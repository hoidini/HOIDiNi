from dataclasses import dataclass
from functools import partial
import os
from typing import List, Optional, Tuple
from copy import deepcopy
from joblib import Memory
import numpy as np
import torch
import rich

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from hoidini.amasstools.geometry import axis_angle_to_matrix
from hoidini.amasstools.smplrifke_feats import SMPLFeatureProcessor
from hoidini.blender_utils.general_blender_utils import blend_scp_and_run
from hoidini.blender_utils.visualize_hoi_animation import visualize_hoi_animation
from hoidini.blender_utils.visualize_stick_figure_blender import visualize_motions
from hoidini.closd.diffusion_planner.data_loaders.get_data import worker_init_fn
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.cphoi.cphoi_utils import smpldata_to_contact_pairs, transform_smpldata
from hoidini.cphoi.samplers.samplers import BaseSampler, InferenceJob, KitchenSampler
from hoidini.datasets.dataset_smplrifke import (
    SamplerWithStartFrame,
    collate_smplrifke_mdm,
    get_cut_by_proximity_range,
    get_mean_and_std,
    load_grab_df_prompts,
    load_hml3d_texts,
)
from hoidini.datasets.grab.grab_utils import (
    get_all_grab_seq_paths,
    get_grab_split_ids,
    grab_seq_path_to_seq_id,
)
from hoidini.datasets.hml3d import get_hml3d_extended_smpldata_lst
from hoidini.datasets.smpldata import SmplData, SmplxFK
from hoidini.datasets.smpldata_preparation import (
    get_extended_smpldata,
    get_grab_extended_smpldata_lst,
)
from hoidini.general_utils import PROJECT_DIR
from hoidini.normalizer import Normalizer
from hoidini.object_conditioning.object_pointcloud_dataset import (
    GrabObjPcdDataset,
    get_grab_point_cloud_dataset,
    get_in_the_wild_object_point_cloud,
)
from hoidini.object_contact_prediction.cpdm_dataset import load_contact_pairs_sequence
from hoidini.resource_paths import GRAB_DATA_PATH
from torch.utils.data import Sampler


class TfmsManager:
    "Helps to retrieve the root transformation in every auto-regressive step"

    def __init__(
        self,
        feature_processor: SMPLFeatureProcessor,
        tfms_processor_lst: List[torch.Tensor],
        normalizer: Optional[Normalizer] = None,
    ):
        self.feature_processor = feature_processor
        self.tfms_processor_lst = tfms_processor_lst
        self.normalizer = normalizer
        self.bs = len(tfms_processor_lst)

    def get_global_tfms_from_features(self, features: torch.Tensor):
        """
        When using with features, it should be the entire sequence length from the beginning of the sequence
        which corresponds to the the tfms_processor which is the initial alignment of the processor
        features: (bs, n_feats, 1, seq_len)
        """
        assert features.shape[0] == self.bs
        tfms_root_global = []
        for b in range(self.bs):
            features_b = (
                features[b].squeeze(1).T
            )  # (n_feats, 1, seq_len) --> (seq_len, n_feats)
            if self.normalizer is not None:
                features_b = self.normalizer.denormalize(features_b)
            smpldata_decoded = self.feature_processor.decode(
                features_b, self.tfms_processor_lst[b]
            )
            tfms_root_global.append(
                get_mat4x4_seq(smpldata_decoded.poses[:, :3], smpldata_decoded.trans)
            )
        return torch.stack(tfms_root_global, dim=0)

    # def get_global_tfms_from_smpldata(self, smpldata: SmplData):
    #     pass


def encode_cphoi_smpldata(smpldata, feature_processor):
    features, tfms_processor = feature_processor.encode(smpldata, return_tfms=True)
    smpldata_decoded = feature_processor.decode(features, tfms_processor[0])
    tfms_root_global = get_mat4x4_seq(
        smpldata_decoded.poses[:, :3], smpldata_decoded.trans
    )
    return features, tfms_processor, smpldata_decoded, tfms_root_global


@dataclass(frozen=True)
class GrabDpDescriptor:
    idx: int
    start_frame: int | None = None
    end_frame: int | None = None
    cache_ind: int | None = None


@dataclass(frozen=True)
class Hml3dDpDescriptor:
    idx: int
    start_frame: int | None = None
    end_frame: int | None = None
    text_ind: int | None = None
    cache_ind: int | None = None


@dataclass(frozen=True)
class MixedDpDescriptor:
    grab_dp: GrabDpDescriptor
    hml3d_dp: Hml3dDpDescriptor
    cache_ind: int


DpDescriptor = GrabDpDescriptor | Hml3dDpDescriptor | MixedDpDescriptor


class PureGrabSampler(Sampler):
    def __init__(self, grab_dp_dicts_len):
        self.grab_dp_dicts_len = grab_dp_dicts_len

    def __iter__(self):
        for i in range(self.grab_dp_dicts_len):
            yield GrabDpDescriptor(i)

    def __len__(self):
        return self.grab_dp_dicts_len


class MixedSampler(Sampler):
    def __init__(
        self,
        n_grab: int,
        n_hml3d: int,
        p_mix: float = 0.3,
        p_pure_grab: float = 0.5,
        n_mixes_lim: int = 10000,
        cache_tgts: Optional[List[str]] = None,
    ):
        if cache_tgts is None:
            cache_tgts = ["mix", "pure_grab"]
        print(
            f"### MixedSampler: p_mix = {p_mix}, p_pure_grab = {p_pure_grab}, p_pure_hml3d = {1-p_mix-p_pure_grab}"
        )
        self.grab_dp_dicts_len = n_grab
        self.hml3d_dp_dicts_len = n_hml3d
        self.opt_lst = ["mix", "pure_grab", "pure_hml3d"]
        assert all(opt in self.opt_lst for opt in cache_tgts)
        self.cache_tgts = cache_tgts
        self.p_lst = [p_mix, p_pure_grab, 1 - p_mix - p_pure_grab]
        assert np.abs(np.sum(self.p_lst) - 1.0) < 1e-6
        self.n_mixes_lim = n_mixes_lim
        self.cached_mixed_outputs = []

    def __len__(self):
        return self.grab_dp_dicts_len

    def get_new_random_dp_descriptor(self, choice: str):
        if choice == "mix":
            result = MixedDpDescriptor(
                GrabDpDescriptor(np.random.randint(self.grab_dp_dicts_len)),
                Hml3dDpDescriptor(np.random.randint(self.hml3d_dp_dicts_len)),
                cache_ind=len(self.cached_mixed_outputs),
            )
        elif choice == "pure_grab":
            result = GrabDpDescriptor(
                np.random.randint(self.grab_dp_dicts_len),
                cache_ind=len(self.cached_mixed_outputs),
            )
        elif choice == "pure_hml3d":
            result = Hml3dDpDescriptor(
                np.random.randint(self.hml3d_dp_dicts_len),
                cache_ind=len(self.cached_mixed_outputs),
            )
        else:
            raise ValueError(f"Invalid choice: {choice}")
        return result

    def __iter__(self):
        for i in range(self.n_mixes_lim):
            choice = np.random.choice(self.opt_lst, p=self.p_lst)
            if (
                choice in self.cache_tgts
                and len(self.cached_mixed_outputs) >= self.n_mixes_lim
            ):
                result = np.random.choice(self.cached_mixed_outputs)
            else:
                result = self.get_new_random_dp_descriptor(choice)
                if choice in self.cache_tgts:
                    self.cached_mixed_outputs.append(result)
            yield result


def apply_xy_plane_aug(smpldata: SmplData) -> SmplData:
    dev, dt = smpldata.poses.device, smpldata.poses.dtype

    def get_random_transform():
        theta = torch.tensor(
            np.random.uniform(0, 2 * np.pi), device=dev, dtype=dt
        )  # scalar
        rot_mat = axis_angle_to_matrix(theta.new_tensor([0.0, 0.0, theta]))  # (3,3)
        # XY translation only (Z stays 0)
        trans_xy = torch.tensor(
            [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0.0],
            device=dev,
            dtype=dt,
        )
        return rot_mat, trans_xy

    rot_mat, trans_xyz = get_random_transform()
    smpldata_new = transform_smpldata(smpldata, rot_mat, trans_xyz)
    return smpldata_new


class CphoiDatasetMixed(Dataset):
    def __init__(
        self,
        grab_dataset_path,
        seq_len=60,
        use_cache=True,
        lim=None,
        cut_grab_by_table_proximity=True,
        features_string="cphoi",
        n_points=1000,
        grab_seq_paths=None,
        grab_split=None,
        pcd_augment_rot_z=False,
        pcd_augment_jitter=False,
        fps=20,
        hml3d_split=None,
        augment_xy_plane_prob=0.0,
        mask_obj_related_features=False,
    ):
        super().__init__()
        self.normalizer: Normalizer | None = None
        self.split = grab_split
        self.grab_dataset_path = grab_dataset_path
        self.seq_len = seq_len
        self.cut_grab_by_table_proximity = cut_grab_by_table_proximity
        self.augment_xy_plane_prob = augment_xy_plane_prob
        self.mask_obj_related_features = mask_obj_related_features
        if grab_seq_paths is None:
            grab_seq_paths = get_all_grab_seq_paths(grab_dataset_path)
            if grab_split is not None:
                split_ids = set(get_grab_split_ids(grab_split))
                grab_seq_paths = [
                    seq_path
                    for seq_path in grab_seq_paths
                    if grab_seq_path_to_seq_id(seq_path) in split_ids
                ]
            if lim is not None:
                grab_seq_paths = list(np.random.permutation(grab_seq_paths))[:lim]

        ################
        # GRAB Point Cloud dataset
        ################
        self.obj_pcd_dataset: GrabObjPcdDataset = get_grab_point_cloud_dataset(
            grab_dataset_path,
            n_points=n_points,
            use_cache=True,
            augment_rot_z=pcd_augment_rot_z,
            augment_jitter=pcd_augment_jitter,
        )

        ################
        # Handle GRAB
        ################
        if use_cache:
            memory = Memory(
                os.path.join(PROJECT_DIR, "cache", "cphoi_extended_smpldata_cache")
            )
            cached_get_extended_smpldata_lst = memory.cache(
                get_grab_extended_smpldata_lst
            )
            ext_grab_smpldata_lst = cached_get_extended_smpldata_lst(
                grab_seq_paths, fk_device=dist_util.dev()
            )
        else:
            ext_grab_smpldata_lst = get_grab_extended_smpldata_lst(
                grab_seq_paths, fk_device=dist_util.dev()
            )

        self.feature_processor = SMPLFeatureProcessor(
            n_pose_joints=ext_grab_smpldata_lst[0]["smpldata"].n_joints,
            n_contact_verts=ext_grab_smpldata_lst[0]["smpldata"].contact.shape[1],
            features_string=features_string,
        )
        self.df_prompts_grab = load_grab_df_prompts()
        grab_dp_dicts = [
            self.get_grab_dp_dict(ext_smpldata)
            for ext_smpldata in ext_grab_smpldata_lst
        ]
        self.grab_dp_dicts = [
            dp_dict
            for dp_dict in grab_dp_dicts
            if len(dp_dict["smpldata"]) >= self.seq_len
        ]

        ################
        # HML3D
        ################
        if hml3d_split is not None:
            if use_cache:
                memory = Memory(
                    os.path.join(
                        PROJECT_DIR, "cache", "cphoi_hml3d_extended_smpldata_cache"
                    )
                )
                cached_get_extended_smpldata_lst = memory.cache(
                    get_hml3d_extended_smpldata_lst
                )
                ext_hml3d_smpldata_lst = cached_get_extended_smpldata_lst(
                    hml3d_split, lim=lim
                )
            else:
                ext_hml3d_smpldata_lst = get_hml3d_extended_smpldata_lst(
                    hml3d_split, lim=lim
                )
            self.hml3d_prompts_dict = {
                esd["hml3d_name"]: load_hml3d_texts(esd["hml3d_name"])
                for esd in ext_hml3d_smpldata_lst
            }
            self.hml3d_dp_dicts = [
                self.get_hml3d_dp_dict(ext_smpldata)
                for ext_smpldata in ext_hml3d_smpldata_lst
            ]
            self.hml3d_dp_dicts = [
                dp_dict
                for dp_dict in self.hml3d_dp_dicts
                if len(dp_dict["smpldata"]) >= self.seq_len
            ]

        self.cached_getitem_results = {}
        self.smplx_fk = SmplxFK(
            seq_len, "cpu"
        )  # TODO: merge motions directly from the features to avoid fk

    def __len__(self):
        return len(self.grab_dp_dicts) + len(self.hml3d_dp_dicts)

    def get_hml3d_dp_dict(self, ext_smpldata):
        smpldata = ext_smpldata["smpldata"]

        # Put zeros in the object related attributes of the hml3d smpldata
        grab_smpldata: SmplData = self.grab_dp_dicts[0]["smpldata"]
        smpldata.local_object_points = torch.zeros(
            len(smpldata), *grab_smpldata.local_object_points.shape[1:]
        )
        smpldata.poses_obj = torch.zeros(
            len(smpldata), *grab_smpldata.poses_obj.shape[1:]
        )
        smpldata.trans_obj = torch.zeros(
            len(smpldata), *grab_smpldata.trans_obj.shape[1:]
        )
        smpldata.contact = torch.zeros(len(smpldata), *grab_smpldata.contact.shape[1:])

        features, tfms_processor, smpldata_decoded, tfms_root_global = (
            encode_cphoi_smpldata(smpldata, self.feature_processor)
        )
        texts = load_hml3d_texts(ext_smpldata["hml3d_name"])
        return {
            "features": features,
            "_smpldata": smpldata,
            "smpldata": smpldata_decoded,  # the decoded version is aligned with the object and the scene together with the smplx results
            "texts": texts,
            "tfms_processor": tfms_processor,
            "tfms_root_global": tfms_root_global,
        }

    def get_grab_dp_dict(self, ext_smpldata):
        seq_path = ext_smpldata["grab_seq_path"]
        smpldata = ext_smpldata["smpldata"]
        object_name = ext_smpldata["object_name"]
        intent_vec = ext_smpldata["intent_vec"]

        # Ugly hack to save some time, the local_object_points should be loaded with the smpldata
        contact_pairs_seq = load_contact_pairs_sequence(seq_path)[0]
        assert len(smpldata) == len(contact_pairs_seq)
        assert len(smpldata.contact) == len(contact_pairs_seq.contacts)
        smpldata.local_object_points = contact_pairs_seq.local_object_points

        if self.cut_grab_by_table_proximity:
            start, end = get_cut_by_proximity_range(smpldata)
            smpldata = smpldata.cut(start, end)

        features, tfms_processor, smpldata_decoded, tfms_root_global = (
            encode_cphoi_smpldata(smpldata, self.feature_processor)
        )
        return {
            "grab_seq_path": seq_path,
            "features": features,
            "_smpldata": smpldata,
            "smpldata": smpldata_decoded,  # the decoded version is aligned with the object and the scene together with the smplx results
            "text": self.df_prompts_grab.loc[grab_seq_path_to_seq_id(seq_path)][
                "Prompt"
            ],
            "object_name": object_name,
            "intent_vec": intent_vec,
            "tfms_processor": tfms_processor,
            "tfms_root_global": tfms_root_global,
        }

    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer = normalizer

    @property
    def n_joints(self):
        return self.feature_processor.n_pose_joints

    @property
    def n_feats(self):
        return sum(self.feature_processor.sizes)

    def get_object_pcd(self, obj_name: str):
        pcd_ds_idx = self.obj_pcd_dataset.name2idx[obj_name]
        return self.obj_pcd_dataset[pcd_ds_idx]

    def __getitem__(self, dp_descriptor: DpDescriptor):

        if dp_descriptor in self.cached_getitem_results:
            return self.cached_getitem_results[dp_descriptor]

        ##############
        # Pure GRAB Datapoint
        ##############
        if isinstance(dp_descriptor, GrabDpDescriptor):
            dp_dict = self.grab_dp_dicts[dp_descriptor.idx]
            if (
                self.augment_xy_plane_prob > 0.0
                and np.random.rand() < self.augment_xy_plane_prob
            ):
                smpldata = dp_dict["_smpldata"]
                smpldata = apply_xy_plane_aug(deepcopy(smpldata))
                features, tfms_processor, _, tfms_root_global = encode_cphoi_smpldata(
                    smpldata, self.feature_processor
                )
            else:
                features = dp_dict["features"]
                tfms_root_global = dp_dict["tfms_root_global"]
                tfms_processor = dp_dict["tfms_processor"]

            if dp_descriptor.start_frame is not None:
                start = dp_descriptor.start_frame
            else:
                start = np.random.randint(len(features) - self.seq_len + 1)
            if dp_descriptor.end_frame is not None:
                end = dp_descriptor.end_frame
            else:
                end = start + self.seq_len

            features = features[start:end]
            tfms_root_global = tfms_root_global[start:end]
            tfm_processor = tfms_processor[0]
            text = dp_dict["text"]
            grab_seq_path = dp_dict["grab_seq_path"]
            object_name = dp_dict["object_name"]
            intent_vec = dp_dict["intent_vec"]
            obj_name = dp_dict["object_name"]
            obj_points = self.get_object_pcd(obj_name)

        ##############
        # Pure HML3D Datapoint
        ##############
        elif isinstance(dp_descriptor, Hml3dDpDescriptor):
            dp_dict = self.hml3d_dp_dicts[dp_descriptor.idx]
            features = dp_dict["features"]
            tfms_root_global = dp_dict["tfms_root_global"]

            if dp_descriptor.start_frame is not None:
                start = dp_descriptor.start_frame
            else:
                start = np.random.randint(len(features) - self.seq_len + 1)
            if dp_descriptor.end_frame is not None:
                end = dp_descriptor.end_frame
            else:
                end = start + self.seq_len

            if dp_descriptor.text_ind is not None:
                text = dp_dict["texts"][dp_descriptor.text_ind]
            else:
                text = np.random.choice(dp_dict["texts"])
            tfms_root_global = tfms_root_global[start:end]
            features = features[start:end]
            tfm_processor = dp_dict["tfms_processor"][0]
            grab_seq_path = None
            obj_points = None
            object_name = None
            intent_vec = None

        ##############
        # Mixed Datapoint
        ##############
        elif isinstance(dp_descriptor, MixedDpDescriptor):
            dp_dict_hml3d = self.hml3d_dp_dicts[dp_descriptor.hml3d_dp.idx]
            dp_dict_grab = self.grab_dp_dicts[dp_descriptor.grab_dp.idx]

            smpldata_hml3d = deepcopy(
                dp_dict_hml3d["smpldata"]
            )  # deepcopy is very important here!
            smpldata_grab = deepcopy(
                dp_dict_grab["smpldata"]
            )  # otherwise the smpldata will be modified in place

            ##############
            # Cut sequences
            ##############
            if dp_descriptor.hml3d_dp.start_frame is not None:
                start_hml3d = dp_descriptor.hml3d_dp.start_frame
            else:
                start_hml3d = np.random.randint(len(smpldata_hml3d) - self.seq_len + 1)
            if dp_descriptor.hml3d_dp.end_frame is not None:
                end_hml3d = dp_descriptor.hml3d_dp.end_frame
            else:
                end_hml3d = start_hml3d + self.seq_len
            if dp_descriptor.grab_dp.start_frame is not None:
                start_grab = dp_descriptor.grab_dp.start_frame
            else:
                start_grab = np.random.randint(len(smpldata_grab) - self.seq_len + 1)
            if dp_descriptor.grab_dp.end_frame is not None:
                end_grab = dp_descriptor.grab_dp.end_frame
            else:
                end_grab = start_grab + self.seq_len

            smpldata_hml3d = smpldata_hml3d.cut(start_hml3d, end_hml3d)
            smpldata_grab = smpldata_grab.cut(start_grab, end_grab)

            ##############
            # Merge motions features
            ##############
            smpldata_merged = deepcopy(smpldata_hml3d)

            poses_merged = smpldata_hml3d.poses.clone()
            poses_merged[:, -30 * 3 :] = smpldata_grab.poses[:, -30 * 3 :]
            smpldata_merged.poses = poses_merged
            with torch.no_grad():
                smplx_out = self.smplx_fk.smpldata_to_smpl_output(
                    smpldata_merged.to(self.smplx_fk.device)
                )
            smpldata_merged.joints = smplx_out.joints
            features, tfms_processor, smpldata, tfms_root_global = (
                encode_cphoi_smpldata(smpldata_merged, self.feature_processor)
            )
            tfm_processor = tfms_processor[0]
            ##############
            # Merge texts
            ##############
            if dp_descriptor.hml3d_dp.text_ind is not None:
                text_hml3d = dp_dict_hml3d["texts"][dp_descriptor.hml3d_dp.text_ind]
            else:
                text_hml3d = np.random.choice(dp_dict_hml3d["texts"])
            text_grab = dp_dict_grab["text"]
            text = f"{text_hml3d} [SEP] {text_grab}"
            obj_points = None
            grab_seq_path = None
            start = None
            end = None
            object_name = None
            intent_vec = None
        else:
            raise ValueError(f"Unknown dp_descriptor: {dp_descriptor}")

        # contact_bits = contact.sum(dim=1) > 0.5
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)

        is_zero_hoi = not isinstance(dp_descriptor, GrabDpDescriptor)
        if is_zero_hoi:
            # zero out the object related features
            mask_obj_related_features = (
                self.feature_processor.get_object_related_feature_mask()
            )  # True where the feature is object related
            features = features * torch.logical_not(mask_obj_related_features)

            # zero out the object points
            obj_points = self.get_object_pcd(
                list(self.obj_pcd_dataset.name2idx.keys())[0]
            )
            obj_points.pos = torch.zeros_like(obj_points.pos)
            obj_points.normal = torch.zeros_like(obj_points.normal)

            # zero out the root transformation which is being used as a condition for the model for better interaction with the object
            tfms_root_global = torch.eye(4).expand(len(features), 4, 4)
        else:
            mask_obj_related_features = torch.zeros(
                features.shape[1], dtype=torch.bool
            )  # (seq_len, ) don't mask anything

            # loss mask

        if "[SEP]" not in text:
            text = f"{text} [SEP] {text}"

        results = {
            "features": features,
            "text": text,
            "obj_points": obj_points,
            "tfms_root_global": tfms_root_global[0],  # (4, 4)
            "tfm_processor": tfm_processor,  # (4, 4)
            "is_zero_hoi_mask": is_zero_hoi,  # Place zeros in the point embeddings
            "metadata": {
                "grab_seq_path": grab_seq_path,
                "object_name": object_name,
                "intent_vec": intent_vec,
                "range": (start, end),
            },
        }
        if self.mask_obj_related_features:
            results["loss_mask"] = mask_obj_related_features.expand(
                *features.shape
            )  # (seq_len, n_features)
        if isinstance(dp_descriptor, (MixedDpDescriptor, GrabDpDescriptor)):
            self.cached_getitem_results[dp_descriptor] = results
        return results


class CphoiDataset(Dataset):
    def __init__(
        self,
        grab_dataset_path,
        seq_len=60,
        use_cache=True,
        lim=None,
        cut_by_table_proximity=True,
        features_string="cphoi",
        n_points=1000,
        grab_seq_paths=None,
        grab_split=None,
        pcd_augment_rot_z=False,
        pcd_augment_jitter=False,
        fps=20,
        augment_xy_plane_prob=0.0,
    ):
        super().__init__()
        self.normalizer: Normalizer | None = None
        self.split = grab_split
        self.grab_dataset_path = grab_dataset_path
        self.seq_len = seq_len
        self.augment_xy_plane_prob = augment_xy_plane_prob
        if grab_seq_paths is None:
            grab_seq_paths = get_all_grab_seq_paths(grab_dataset_path)
            if grab_split is not None:
                split_ids = set(get_grab_split_ids(grab_split))
                grab_seq_paths = [
                    seq_path
                    for seq_path in grab_seq_paths
                    if grab_seq_path_to_seq_id(seq_path) in split_ids
                ]
            if lim is not None:
                grab_seq_paths = list(np.random.permutation(grab_seq_paths))[:lim]
        self.obj_pcd_dataset: GrabObjPcdDataset = get_grab_point_cloud_dataset(
            grab_dataset_path,
            n_points=n_points,
            use_cache=True,
            augment_rot_z=pcd_augment_rot_z,
            augment_jitter=pcd_augment_jitter,
        )
        if use_cache:
            memory = Memory(os.path.join(PROJECT_DIR, "cphoi_extended_smpldata_cache"))
            cached_get_extended_smpldata_lst = memory.cache(
                get_grab_extended_smpldata_lst
            )
            ext_smpldata_lst = cached_get_extended_smpldata_lst(
                grab_seq_paths, fk_device=dist_util.dev()
            )
        else:
            ext_smpldata_lst = get_grab_extended_smpldata_lst(
                grab_seq_paths, fk_device=dist_util.dev()
            )

        self.feature_processor = SMPLFeatureProcessor(
            n_pose_joints=ext_smpldata_lst[0]["smpldata"].n_joints,
            n_contact_verts=ext_smpldata_lst[0]["smpldata"].contact.shape[1],
            features_string=features_string,
        )

        df_prompts = load_grab_df_prompts()

        self.datapoints = []
        for ext_smpldata in tqdm(
            ext_smpldata_lst, total=len(ext_smpldata_lst), desc="Loading CphoiDataset"
        ):
            seq_path = ext_smpldata["grab_seq_path"]
            smpldata = ext_smpldata["smpldata"]
            object_name = ext_smpldata["object_name"]
            intent_vec = ext_smpldata["intent_vec"]

            # Ugly hack to save some time, the local_object_points should be loaded with the smpldata
            contact_pairs_seq = load_contact_pairs_sequence(seq_path)[0]
            assert len(smpldata) == len(contact_pairs_seq)
            assert len(smpldata.contact) == len(contact_pairs_seq.contacts)
            smpldata.local_object_points = contact_pairs_seq.local_object_points

            if lim is not None and len(self.datapoints) == lim:
                break

            if cut_by_table_proximity:
                start, end = get_cut_by_proximity_range(smpldata)
                smpldata = smpldata.cut(start, end)

            if len(smpldata) < self.seq_len:
                continue

            features, tfms_processor, smpldata_decoded, tfms_root_global = (
                encode_cphoi_smpldata(smpldata, self.feature_processor)
            )
            self.datapoints.append(
                {
                    "grab_seq_path": seq_path,
                    "features": features,
                    "_smpldata": smpldata,
                    "smpldata": smpldata_decoded,  # the decoded version is aligned with the object and the scene together with the smplx results
                    "text": df_prompts.loc[grab_seq_path_to_seq_id(seq_path)]["Prompt"],
                    "object_name": object_name,
                    "intent_vec": intent_vec,
                    "tfms_processor": tfms_processor,
                    "tfms_root_global": tfms_root_global,
                }
            )

    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer = normalizer

    @property
    def n_joints(self):
        return self.feature_processor.n_pose_joints

    @property
    def n_feats(self):
        return sum(self.feature_processor.sizes)

    def __len__(self):
        return len(self.datapoints)

    def get_object_pcd(self, obj_name: str):
        pcd_ds_idx = self.obj_pcd_dataset.name2idx[obj_name]
        return self.obj_pcd_dataset[pcd_ds_idx]

    def __getitem__(self, idx: int | Tuple[int, int]):
        if isinstance(idx, tuple):
            idx, start_frame = idx
        else:
            start_frame = None

        data = self.datapoints[idx]
        if (
            self.augment_xy_plane_prob > 0.0
            and np.random.rand() < self.augment_xy_plane_prob
        ):
            smpldata = data["_smpldata"]
            smpldata = apply_xy_plane_aug(deepcopy(smpldata))
            features_seq, tfms_processor, smpldata, tfms_root_global = (
                encode_cphoi_smpldata(smpldata, self.feature_processor)
            )
        else:
            features_seq = data["features"]
            tfms_root_global = data["tfms_root_global"]
            tfms_processor = data["tfms_processor"]

        if start_frame is not None:
            start = start_frame
        else:
            start = np.random.randint(len(features_seq) - self.seq_len + 1)
        end = start + self.seq_len
        features = features_seq[start:end]
        tfms_root_global = tfms_root_global[start]
        tfm_processor = tfms_processor[start]

        if self.normalizer is not None:
            features = self.normalizer.normalize(features)

        obj_name = data["object_name"]
        obj_points = self.get_object_pcd(obj_name)

        """
        tfm_processor stores the processor alignment and it's being used to cancel it for inference time
            it's a (4, 4) matrix
        tfms_root_global is root rotation in each frame (after the alignment fix!!!)
            it's a (seq_len, 4, 4) matrix
        So, to sum up, tfm_processor should be used with the when the features are being decoded to cancel the processor alignment
        and then, tfms_root_global should be inserted to the model's condition to inform it about the actual root transformation
        """

        return {
            "features": features,
            "text": data["text"],
            "obj_points": obj_points,
            "tfms_root_global": tfms_root_global,  # (4, 4)
            "tfm_processor": tfm_processor,  # (4, 4)
            "metadata": {
                "grab_seq_path": data["grab_seq_path"],
                "object_name": data["object_name"],
                "intent_vec": data["intent_vec"],
                "range": (start, end),
            },
        }


def get_mat4x4(rot_axis_angle, trans):
    rot_mat = axis_angle_to_matrix(rot_axis_angle)
    cur_tfm = torch.eye(4, device=rot_mat.device)
    cur_tfm[:3, :3] = rot_mat
    cur_tfm[:3, 3] = trans
    return cur_tfm


def get_mat4x4_seq(rot_axis_angle, trans):
    """
    rot_axis_angle: (T, 3)
    trans: (T, 3)
    """
    rot_mat = axis_angle_to_matrix(rot_axis_angle)
    tfm = torch.eye(4, device=rot_mat.device).repeat(rot_mat.shape[0], 1, 1)
    tfm[:, :3, :3] = rot_mat
    tfm[:, :3, 3] = trans
    return tfm


def get_cphoi_dataloader(
    grab_dataset_path: str,
    batch_size: int,
    experiment_dir: str,
    is_training: bool,
    context_len: int = 20,
    pred_len: int = 40,
    lim: Optional[int] = None,
    n_points: int = 512,
    seed=None,
    use_normalizer: bool = True,
    grab_seq_paths: Optional[List[str]] = None,
    pcd_augment_rot_z: bool = False,
    pcd_augment_jitter: bool = False,
    fps: int = 20,
    feature_names: Optional[List[str] | str] = None,
    grab_split: Optional[str] = "train",
    pred_len_dataset: Optional[int] = None,
    is_enforce_motion_length: bool = False,
    mixed_dataset: bool = False,
    augment_xy_plane_prob: bool = False,
    mask_obj_related_features: bool = False,
):
    if pred_len_dataset is None:
        pred_len_dataset = pred_len
    total_seq_len_ds = context_len + pred_len_dataset
    total_seq_len_model = context_len + pred_len

    assert fps == 20

    if not mixed_dataset:
        dataset = CphoiDataset(
            grab_dataset_path,
            seq_len=total_seq_len_ds,
            lim=lim,
            n_points=n_points,
            grab_seq_paths=grab_seq_paths,
            pcd_augment_rot_z=pcd_augment_rot_z,
            pcd_augment_jitter=pcd_augment_jitter,
            fps=fps,
            features_string=feature_names,
            grab_split=grab_split,
            use_cache=True,
            augment_xy_plane_prob=augment_xy_plane_prob,
        )
        sampler = None
    else:
        dataset = CphoiDatasetMixed(
            grab_dataset_path,
            features_string=feature_names,
            use_cache=True,
            cut_grab_by_table_proximity=False,
            hml3d_split="train",
            lim=lim,
            pcd_augment_rot_z=pcd_augment_rot_z,
            pcd_augment_jitter=pcd_augment_jitter,
            fps=fps,
            grab_split=grab_split,
            augment_xy_plane_prob=augment_xy_plane_prob,
            mask_obj_related_features=mask_obj_related_features,
        )
        sampler = MixedSampler(
            n_grab=len(dataset.grab_dp_dicts),
            n_hml3d=len(dataset.hml3d_dp_dicts),
            p_mix=0.3,
            p_pure_grab=0.5,
            n_mixes_lim=2000,
            cache_tgts=["mix", "pure_grab"],
        )

    enforce_motion_length = None
    if is_enforce_motion_length:  # :(
        enforce_motion_length = total_seq_len_model
    collate = partial(
        collate_smplrifke_mdm,
        pred_len=pred_len,
        context_len=context_len,
        enforce_motion_length=enforce_motion_length,
    )
    if sampler is None:
        shuffle = seed is None or grab_seq_paths is not None
    else:
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
        collate_fn=collate,
        shuffle=shuffle,
        persistent_workers=True,
        worker_init_fn=partial(worker_init_fn, seed=seed),
        sampler=sampler,
    )

    if use_normalizer:
        if is_training:
            mean, std = get_mean_and_std(dataloader, sample_size=2000)
            normalizer = Normalizer(mean, std, eps=1e-5)
            normalizer.save(experiment_dir)
        else:
            normalizer = Normalizer.from_dir(experiment_dir)
        # dataset.set_normalizer(normalizer)
        dataset.set_normalizer(normalizer)

        # initialize the dataloader with the new version of the dataset that has the normalizer
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate,
            shuffle=shuffle,
            persistent_workers=True,
            worker_init_fn=partial(worker_init_fn, seed=seed),
            sampler=sampler,
        )
    print("Normalizer is not None", dataloader.dataset.normalizer is not None)
    return dataloader


def main():
    features_string = "cphoi"
    dataset = CphoiDataset(
        grab_dataset_path=GRAB_DATA_PATH,
        lim=2,
        features_string=features_string,
        use_cache=False,
        cut_by_table_proximity=False,
    )

    dp_idx = 0

    # smpldata = dataset.datapoints[dp_idx]["smpldata_decoded"]
    dp = dataset[(dp_idx, 0)]
    smpldata = dataset.datapoints[dp_idx]["smpldata"].cut(0, dataset.seq_len)
    tfms_root_global = dp["tfms_root_global"]
    tfm_processor = dp["tfm_processor"]
    features = dp["features"]
    smpldata_decoded = dataset.feature_processor.decode(features, tfm_processor)
    # smpldata_decoded = dataset.feature_processor.decode(features)

    visualize_hoi_animation(
        [smpldata, smpldata_decoded],
        mat_lst=tfms_root_global[::15],
        object_path_or_name=dataset.datapoints[dp_idx]["object_name"],
        text=dataset.datapoints[dp_idx]["text"],
        start_frame=0,
        smplx_cancel_offset=True,
    )


def main_test_dataloader():
    dataloader = get_cphoi_dataloader(
        grab_dataset_path=GRAB_DATA_PATH,
        batch_size=2,
        experiment_dir="/home/dcor/roeyron/tmp/cphoi_test",
        is_training=True,
        context_len=20,
        feature_names="cphoi",
        pred_len=40,
        lim=2,
        n_points=512,
        seed=42,
        use_normalizer=True,
    )

    for motion, cond in dataloader:
        print(motion)
        print(cond)
        break

    assert "tfms_root_global" in cond["y"]
    print(cond["y"]["tfms_root_global"].shape)
    assert "tfm_processor" in cond["y"]
    print(cond["y"]["tfm_processor"].shape)


class LazyInferenceDataset:
    def __init__(
        self,
        grab_dataset_path: str,
        n_points: int,
        normalizer: Normalizer,
        features_string: str,
        mixed_dataset: bool = False,
    ):
        self.grab_dataset_path = grab_dataset_path
        self.obj_pcd_dataset = get_grab_point_cloud_dataset(
            grab_dataset_path,
            n_points=n_points,
            use_cache=True,
            augment_rot_z=False,
            augment_jitter=False,
        )
        self.mixed_dataset = mixed_dataset
        self.feature_processor = SMPLFeatureProcessor(
            n_pose_joints=52,
            n_contact_verts=60,
            features_string=features_string,
        )
        self.df_prompts = load_grab_df_prompts()
        self.normalizer = normalizer

    def get_object_pcd(self, obj_name: str):
        if os.path.exists(obj_name):
            print(f"Loading object from {obj_name}")
            return get_in_the_wild_object_point_cloud(obj_name)
        else:
            pcd_ds_idx = self.obj_pcd_dataset.name2idx[obj_name]
            return self.obj_pcd_dataset[pcd_ds_idx]

    def get_datapoint(
        self, inference_job: InferenceJob, sampler: Optional[BaseSampler] = None
    ):
        grab_seq_path = inference_job.grab_seq_path or np.random.choice(
            get_all_grab_seq_paths(self.grab_dataset_path)
        )
        print(f"LazyInferenceDataset: Loading grab sequence from {grab_seq_path}")
        ext_smpldata = get_extended_smpldata(grab_seq_path)
        smpldata = ext_smpldata["smpldata"]
        intent_vec = ext_smpldata["intent_vec"]
        object_name = ext_smpldata["object_name"]

        # TODO: do it inside the get_extended_smpldata function
        contact_pairs_seq, _ = load_contact_pairs_sequence(grab_seq_path)
        assert len(smpldata) == len(contact_pairs_seq)
        assert len(smpldata.contact) == len(contact_pairs_seq.contacts)
        smpldata.local_object_points = contact_pairs_seq.local_object_points

        # set text
        if inference_job.text is not None:
            text = inference_job.text
        else:
            text = self.df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)]["Prompt"]

        if self.mixed_dataset and "[SEP]" not in text:
            text = f"{text} [SEP] {text}"

        # set object
        if inference_job.object_name is not None:
            object_name = inference_job.object_name

        if sampler is not None and isinstance(sampler, KitchenSampler):
            _start, _end, _obj_orient, _obj_transl = sampler.get_prefix_values()
            smpldata.poses_obj[_start:_end] = _obj_orient
            smpldata.trans_obj[_start:_end] = _obj_transl

        # set start
        if inference_job.start_frame is not None:
            start = inference_job.start_frame
        else:
            start, _ = get_cut_by_proximity_range(smpldata)
            assert sampler is None or not isinstance(sampler, KitchenSampler)

        # features, tfms_processor = self.feature_processor.encode(smpldata, return_tfms=True)
        # smpldata_decoded = self.feature_processor.decode(features, tfms_processor[0])
        # tfms_root_global = get_mat4x4_seq(smpldata_decoded.poses[:, :3], smpldata_decoded.trans)
        features, tfms_processor, smpldata_decoded, tfms_root_global = (
            encode_cphoi_smpldata(smpldata, self.feature_processor)
        )

        features = features[start:]
        tfms_root_global = tfms_root_global[start:]

        if self.normalizer is not None:
            features = self.normalizer.normalize(features)

        obj_points = self.get_object_pcd(object_name)

        return {
            "features": features,
            "text": text,
            "obj_points": obj_points,
            "tfms_root_global": tfms_root_global,  # (seq_len, 4, 4)
            "tfm_processor": tfms_processor[0],  # (4, 4)
            "metadata": {
                "grab_seq_path": grab_seq_path,
                "intent_vec": intent_vec,
                "object_name": object_name,
                "range": (start, None),
            },
        }

    # main_test_dataloader()


def main_test_mixed_dataset_and_sampler():
    is_mixed = True
    hml3d_split = "train" if is_mixed else None

    dataset = CphoiDatasetMixed(
        grab_dataset_path=GRAB_DATA_PATH,
        features_string="cphoi",
        use_cache=False,
        cut_grab_by_table_proximity=False,
        hml3d_split=hml3d_split,
        lim=10,
    )
    if is_mixed:
        sampler = MixedSampler(
            n_grab=len(dataset.grab_dp_dicts),
            n_hml3d=len(dataset.hml3d_dp_dicts),
            p_mix=0.3,
            p_pure_grab=0.5,
            n_mixes_lim=10000,
        )
    else:
        sampler = PureGrabSampler(len(dataset.grab_dp_dicts))

    collate = partial(collate_smplrifke_mdm, pred_len=40, context_len=20)
    dataloader = DataLoader(
        dataset, batch_size=2, collate_fn=collate, shuffle=False, sampler=sampler
    )
    for batch in dataloader:
        print(batch)
        break


def main_mixed():
    # Mixed dataset
    rich.print("[red]Mixed dataset[/red]")
    dataset = CphoiDatasetMixed(
        grab_dataset_path=GRAB_DATA_PATH,
        features_string="cphoi",
        cut_grab_by_table_proximity=False,
        seq_len=100,
        hml3d_split="train",
        lim=5,
        augment_xy_plane_prob=1.0,
    )
    grab_dp_descriptor = GrabDpDescriptor(idx=0, start_frame=0)
    hml3d_dp_descriptor = Hml3dDpDescriptor(idx=2, start_frame=0)
    mixed_dp_descriptor = MixedDpDescriptor(
        grab_dp=grab_dp_descriptor, hml3d_dp=hml3d_dp_descriptor, cache_ind=0
    )
    dp_grab = dataset[grab_dp_descriptor]
    smpldata_grab = dataset.feature_processor.decode(
        dp_grab["features"], dp_grab["tfm_processor"]
    )
    dp_hml3d = dataset[hml3d_dp_descriptor]
    smpldata_hml3d = dataset.feature_processor.decode(
        dp_hml3d["features"], dp_hml3d["tfm_processor"]
    )
    dp_mixed = dataset[mixed_dp_descriptor]
    smpldata_mixed = dataset.feature_processor.decode(
        dp_mixed["features"], dp_mixed["tfm_processor"]
    )
    save_path = "/home/dcor/roeyron/tmp/cphoi_data_test.blend"
    cp_grab = smpldata_to_contact_pairs(smpldata_grab)
    visualize_hoi_animation(
        [smpldata_grab, smpldata_hml3d, smpldata_mixed],
        save_path=save_path,
        mat_lst=[
            dp_grab["tfms_root_global"],
            dp_hml3d["tfms_root_global"],
            dp_mixed["tfms_root_global"],
        ],
        contact_pairs_seq=cp_grab,
        object_path_or_name=dp_grab["metadata"]["object_name"],
        visualize_from_joints_as_well=True,
    )


def main_regular():
    # Main dataset
    augment_xy_plane_prob = 1.0
    rich.print("[red]Main dataset[/red]")
    use_cache = True
    dataset = CphoiDataset(
        grab_dataset_path=GRAB_DATA_PATH,
        features_string="cphoi",
        use_cache=use_cache,
        cut_by_table_proximity=False,
        augment_xy_plane_prob=augment_xy_plane_prob,
        lim=10,
    )
    dp = dataset[5]
    features = dp["features"]
    smpldata0 = dataset.feature_processor.decode(features, dp["tfm_processor"])
    save_path = "/home/dcor/roeyron/tmp/cphoi_data_test.blend"
    visualize_motions([smpldata0.joints], save_path=save_path)


def main_lazy():

    # Lazy inference dataset
    rich.print("[red]Lazy inference dataset[/red]")
    inference_job = InferenceJob(start_frame=0, text=None)
    lazy_dataset = LazyInferenceDataset(
        grab_dataset_path=GRAB_DATA_PATH,
        n_points=512,
        normalizer=None,
        features_string="cphoi",
        mixed_dataset=True,
    )
    dp = lazy_dataset.get_datapoint(inference_job)
    features = dp["features"]
    smpldata2 = lazy_dataset.feature_processor.decode(features, dp["tfm_processor"])
    save_path = "/home/dcor/roeyron/tmp/cphoi_data_test.blend"
    visualize_motions([smpldata2.joints], save_path=save_path)
    blend_scp_and_run(save_path)


if __name__ == "__main__":
    main_mixed()
    # main_regular()
    # main_test_mixed_dataset_and_sampler()
