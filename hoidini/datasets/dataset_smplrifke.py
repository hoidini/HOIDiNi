from __future__ import annotations
from collections import defaultdict
from glob import glob
from abc import ABC
import pickle
import pandas as pd
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from functools import partial
from typing import Callable, Optional, Set, Tuple, List
import numpy as np
from tqdm import tqdm
from hoidini.amasstools.smplrifke_feats import SMPLFeatureProcessor
from hoidini.datasets.grab.grab_utils import (
    get_all_grab_seq_paths,
    get_grab_split_ids,
    grab_seq_path_to_seq_id,
)
from hoidini.datasets.smpldata import SmplData
from hoidini.datasets.smpldata_preparation import get_grab_extended_smpldata_lst
from hoidini.general_utils import PROJECT_DIR, SRC_DIR
from hoidini.normalizer import Normalizer
from hoidini.closd.diffusion_planner.data_loaders.tensors import collate
import random
from torch.utils.data import Sampler
import math
from hoidini.object_conditioning.object_pointcloud_dataset import (
    GrabObjPcdDataset,
    get_grab_point_cloud_dataset,
    PyGData,
)
from hoidini.resource_paths import GRAB_DATA_PATH, HUMANML3D_DATASET_PATH
from hoidini.datasets.smpldata import SmplxFK
from joblib import Memory
from hoidini.closd.diffusion_planner.utils import dist_util


class SamplerWithStartFrame(Sampler):
    def __init__(self, epoch_size: int, start_frame_dist: Optional[Callable] = None):
        self.epoch_size = epoch_size
        if start_frame_dist is None:

            def default_start_frame():
                return 0

            start_frame_dist = default_start_frame
        self.start_frame_dist = start_frame_dist

    def __iter__(self):
        for i in range(self.epoch_size):
            yield i, self.start_frame_dist()

    def __len__(self):
        return self.epoch_size


def samples_to_smpldata_lst(
    sample_raw: torch.Tensor,
    normalizer: Normalizer,
    feature_processor: SMPLFeatureProcessor,
    tfms_processor_lst: Optional[List[torch.Tensor]] = None,
) -> List[SmplData]:
    # sample_raw.shape = (batch_size, 217, 1, seq_len)
    sample_raw_reshaped = sample_raw[:, :, 0, :].permute(
        0, 2, 1
    )  # -> (batch, seq_len, n_features)
    sample_normalized = normalizer.denormalize(
        sample_raw_reshaped
    )  # (batch, seq_len, n_features)
    smpldata_lst = []
    for i in range(sample_normalized.shape[0]):
        f = sample_normalized[i]
        tfm_processor = (
            tfms_processor_lst[i] if tfms_processor_lst is not None else None
        )
        smpldata = feature_processor.decode(f, tfm_processor)
        smpldata_lst.append(smpldata)
    return smpldata_lst


def load_hml3d_texts(dp_id):
    text_path = os.path.join(HUMANML3D_DATASET_PATH, "texts", f"{dp_id}.txt")
    with open(text_path, "r") as f:
        texts = f.read().split("\n")
    texts = [t.split("#")[0] for t in texts]
    texts = [t for t in texts if t != ""]
    return texts


def load_grab_df_prompts():
    df_prompts = pd.read_csv(
        os.path.join(SRC_DIR, "datasets/resources/grab_prompts.csv")
    )
    df_prompts["dp_ind"] = df_prompts["dp_ind"].apply(lambda n: n.replace(".npz", ""))
    df_prompts = df_prompts.set_index("dp_ind")
    return df_prompts


def load_hml_split_ids(split) -> Set[str]:
    with open(os.path.join(HUMANML3D_DATASET_PATH, f"{split}.txt"), "r") as f:
        split_ids = set(f.read().split("\n"))
    return set(split_ids)


def get_mean_and_std0(
    dataset: SmplRifkeDataset, sample_size: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    assert dataset.normalizer is None
    if sample_size is not None:
        inds = np.random.choice(
            len(dataset), size=min(len(dataset), sample_size), replace=False
        )
    else:
        inds = list(range(len(dataset)))
    features_lst = []
    for i in tqdm(inds, desc="Calc mean and std"):
        dp = dataset[i]
        features = dp[0] if isinstance(dp, tuple) else dp["features"]
        features_lst.append(features)
    features = torch.concatenate(features_lst)
    return features.mean(0), features.std(0)


def get_mean_and_std(
    dataloader: DataLoader, sample_size: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    assert dataloader.dataset.normalizer is None
    n_samples = 0
    features_lst = []
    for features, _ in dataloader:
        n_samples += features.shape[0]
        features = features.squeeze(2).permute(
            0, 2, 1
        )  # (bs, n_feats, 1, seq) -> (bs, seq, n_feats)
        features = features.reshape(-1, features.shape[-1])
        features_lst.append(features)
        if n_samples >= sample_size:
            break
    features = torch.concatenate(features_lst, dim=0)
    return features.mean(0), features.std(0)


def collate_smplrifke_mdm(
    batch: list, pred_len: int, context_len: int, enforce_motion_length: int = None
):
    """
    batch could be:
        - a list of tuples (features, text)
        - a list of dicts with keys "features" and "text"
        - a list of dicts with keys "features", "text", and "obj_points"
    """
    if isinstance(batch[0], tuple):
        batch = [{"features": f, "text": t} for f, t in batch]

    adapted_batch = []
    for data_point in batch:
        features = data_point["features"]
        if enforce_motion_length and features.shape[0] < enforce_motion_length:
            # Use only in inference mode when we don't care about the motion itself and we only want the prefix
            features_new = torch.zeros(
                enforce_motion_length,
                features.shape[1],
                device=features.device,
                dtype=features.dtype,
            )
            features_new[: features.shape[0]] = features
            features = features_new
        d = {
            "inp": features.T.float().unsqueeze(1)[
                ..., context_len:
            ],  # [seq_len, J] -> [J, 1, seq_len]
            "prefix": data_point["features"].T.float().unsqueeze(1)[..., :context_len],
            "text": data_point["text"],
            "lengths": pred_len,
        }
        if "obj_points" in data_point:
            d["obj_points"] = data_point["obj_points"]
        if "metadata" in data_point:
            d["metadata"] = data_point["metadata"]
        if "condition_mask" in data_point:
            d["condition_mask"] = (
                data_point["condition_mask"].T.float().unsqueeze(1)[..., context_len:]
            )
            d["condition_input"] = (
                data_point["condition_input"].T.float().unsqueeze(1)[..., context_len:]
            )
        if "tfms_root_global" in data_point:
            d["tfms_root_global"] = data_point["tfms_root_global"]
        if "tfm_processor" in data_point:
            d["tfm_processor"] = data_point["tfm_processor"]
        if "is_zero_hoi_mask" in data_point:
            d["is_zero_hoi_mask"] = data_point["is_zero_hoi_mask"]  # boolean
        if "loss_mask" in data_point:
            d["loss_mask"] = (
                data_point["loss_mask"].T.float().unsqueeze(1)[..., context_len:]
            )  # (n_features, seq_len)
        adapted_batch.append(d)
    return collate(adapted_batch)


class SmplRifkeDataset(Dataset, ABC):

    smpl_model_name = None

    def __init__(self):
        super().__init__()
        self.normalizer: Normalizer | None = None
        self.smpl_data_lst: List[SmplData] = []

    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer = normalizer

    @property
    def n_joints(self):
        return self.smpl_data_lst[0].joints.shape[1]

    @property
    def n_feats(self):
        return self.normalizer.n_feats

    def __len__(self):
        return len(self.smpl_data_lst)


class GrabRifkeDataset(SmplRifkeDataset):
    smpl_model_name = "smplx"

    def __init__(
        self, dataset_path=None, seq_len=60, lim=None, cut_by_table_proximity=True
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.seq_len = seq_len

        all_seq_paths = sorted(glob("s*/*.npz", root_dir=dataset_path))
        all_seq_paths = [os.path.join(dataset_path, rp) for rp in all_seq_paths]

        df_prompts = load_grab_df_prompts()

        self.feature_lst = []
        self.smpl_data_lst = []
        self.text_lst = []
        for seq_path in tqdm(all_seq_paths):
            dp_id = os.path.relpath(seq_path, dataset_path)
            if lim is not None and len(self.feature_lst) == lim:
                break
            smpldata = SmplData.load(seq_path)

            if cut_by_table_proximity:
                start, end = get_cut_by_proximity_range(smpldata)
                smpldata = smpldata.cut(start, end)
            if len(smpldata) < self.seq_len:
                continue

            features = smpldata.to_smplrifke_features()
            self.feature_lst.append(features)
            self.smpl_data_lst.append(smpldata)
            self.text_lst.append(df_prompts.loc[dp_id.replace(".npz", "")].Prompt)

    def __getitem__(self, idx):
        features_seq = self.feature_lst[idx]
        text = self.text_lst[idx]
        start = np.random.randint(len(features_seq) - self.seq_len + 1)
        end = start + self.seq_len
        features = features_seq[start:end]
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
        return features, text


class HML3DSmplRifkeDataset(SmplRifkeDataset):

    smpl_model_name = "smpl"
    """
    Should work for body/fullbody/hands
    """

    def __init__(self, dataset_path, seq_len=16, split="train", lim=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.seq_len = seq_len

        dp_ids = sorted(
            [
                dp_id.split(".npz")[0]
                for dp_id in os.listdir(dataset_path)
                if dp_id.endswith(".npz")
            ]
        )
        if split is not None and split != "":
            split_ids = load_hml_split_ids(split)
            dp_ids = [dp_id for dp_id in dp_ids if dp_id in split_ids]

        self.feature_lst = []
        self.smpl_data_lst = []
        self.texts_lst = []
        for dp_id in tqdm(dp_ids):
            if lim is not None and len(self.feature_lst) == lim:
                break
            dp_path = os.path.join(dataset_path, f"{dp_id}.npz")
            smpl_data = SmplData.load(dp_path)
            if len(smpl_data) < self.seq_len:
                continue
            features = smpl_data.to_smplrifke_features()
            self.feature_lst.append(features)
            self.smpl_data_lst.append(smpl_data)
            texts = load_hml3d_texts(dp_id)
            self.texts_lst.append(texts)

    def __getitem__(self, idx):
        features_seq = self.feature_lst[idx]
        texts = self.texts_lst[idx]
        text = np.random.choice(texts)
        start = np.random.randint(len(features_seq) - self.seq_len + 1)
        end = start + self.seq_len
        features = features_seq[start:end]
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
        return features, text


def get_cut_by_proximity_range(smpl_data: SmplData, th_rel=0.85) -> Tuple[int, int]:
    """
    Drop frames further away from the th_rel
    """
    y = smpl_data.joints[:, 0, 1]
    relevant_inds = torch.where(y < (y.max() * th_rel + y.min() * (1 - th_rel)))[0]
    start = relevant_inds[0]
    end = relevant_inds[-1]
    return start, end


class MixedSampler(Sampler):
    def __init__(self, A_indices, B_indices, p_B_pair, epoch_size, output_limit=None):
        """
        Args:
            A_indices (list): List of indices corresponding to set A.
            B_indices (list): List of indices corresponding to subset B.
            p_B_pair (float): Fraction of samples (between 0 and 1) where a == b (both from B).
            num_samples (int): Total number of samples to generate at each epoch.
            output_limit optional[int]: useful when the dataset is using caching
        """
        self.A_indices = A_indices
        self.B_indices = B_indices
        self.p_B_pair = p_B_pair
        self.epoch_size = epoch_size
        self.output_limit = output_limit
        self.cached_outputs = []
        self.output_id = 0

    def __iter__(self):
        print(
            f"### Sampler is starting a new epoch, len(cached_outputs) = {len(self.cached_outputs)} / {self.output_limit}"
        )
        for i in range(self.epoch_size):
            if self.output_limit and len(self.cached_outputs) >= self.output_limit:
                result = self.cached_outputs[np.random.choice(len(self.cached_outputs))]
                if i == 0:
                    print("### Sampler is using cached outputs")
            else:
                if np.random.uniform() < self.p_B_pair:
                    # p_B_pair% chance: choose a matching pair from B
                    idx = random.choice(self.B_indices)
                    a_idx, b_idx = idx, idx
                else:
                    # Otherwise: choose a from A and b from B (they might be different)
                    a_idx = random.choice(self.A_indices)
                    b_idx = random.choice(self.B_indices)
                if i == 0:
                    print("### Sampler is using sampling new outputs")

                result = (a_idx, b_idx, self.output_id)
                self.output_id += 1
                if self.output_limit is not None:
                    self.cached_outputs.append(result)

            yield result

    def __len__(self):
        return self.epoch_size


def get_grab_amass_datapoint_name(dp_name, df_grab_prompts) -> Optional[str]:
    """
    There is a mismatch between the GRAB dataset names and the AMASS-GRAB dataset names.
    This function maps the GRAB names to the AMASS-GRAB names.
    """
    if dp_name not in df_grab_prompts.index:
        dp_name = dp_name.replace("pick_all", "lift")
    if dp_name in [
        "s5/banana_eat_1",
        "s5/phone_lift",
        "s5/cylindersmall_lift",
        "s5/airplane_fly_1",
        "s3/multi_bottle_whineglass_1",
    ]:
        print(f"### {dp_name} not found in df_grab_prompts")
        return None
    assert dp_name in df_grab_prompts.index
    return dp_name


class MixedDataset(SmplRifkeDataset):
    """
    A newer version of the Hml3DGrabDataset, where the body and hands are taken from different sources.
    The difference is that Grab is taken directly from the GRAB dataset, instead of Amass since it has no contact annotations there.
    """

    def __init__(
        self,
        dataset_path,
        seq_len=60,
        hml_split="train",
        lim=None,
        fk_device="cpu",
        use_cache=True,
        grab_seq_paths=None,
        grab_split=None,
        features_string: str = "human",
        condition_input_feature: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        ddl = defaultdict(list)


class Hml3DGrabDataset(SmplRifkeDataset):
    """
    The 'hml3d_grab' dataset is composed from merged motions from HumanML3D (a subset of it) and the GRAB
    dataset, where for each motion, the body is taken from HumanML3D+GRAB and the hands are taken from GRAB
    The returned text is a concatenation of the texts of the two source motions.
    """

    smpl_model_name = "smplx"

    def __init__(
        self,
        dataset_path,
        seq_len=60,
        hml_split="train",
        lim=None,
        fk_device="cpu",
        use_cache=True,
        grab_seq_paths=None,
        grab_split=None,
        features_string: str = "human",
        condition_input_feature: Optional[str] = None,
    ):
        """
        condition_input: if provided, will use it's value to query the feature_processor to get the condition mask
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.condition_input_feature = condition_input_feature
        self.smplx_fk = SmplxFK(
            seq_len, fk_device
        )  # TODO: merge motions directly from the features to avoid fk
        self.use_cache = use_cache
        self.cache = {}

        self.feature_processor = SMPLFeatureProcessor(features_string=features_string)

        # prepare dataframe
        df = pd.read_csv(os.path.join(dataset_path, "index.csv"))
        df = df.drop(columns="Unnamed: 0")
        df["path"] = df.new_name.apply(
            lambda new_name: os.path.join(
                dataset_path, new_name.replace(".npy", ".npz")
            )
        )
        df["dp_name"] = df["new_name"].apply(
            lambda name: name.replace(".npy", "").replace(".npz", "")
        )  # dp name w/o file extension

        # filter by HumanML3D split
        if hml_split is not None and hml_split != "":
            hml_split_ids = set(load_hml_split_ids(hml_split))
            df = df[
                (df["dataset"] == "GRAB")
                | df["dp_name"].apply(lambda n: n in hml_split_ids)
            ]  # GRAB isn't part of HumanML3D

        # filter by GRAB split
        if grab_seq_paths is None and grab_split is not None:
            grab_split_ids = set(get_grab_split_ids(grab_split))
            df = df[
                (df["dataset"] != "GRAB")
                | df["dp_name"].apply(lambda n: n in grab_split_ids)
            ]

        df_grab_prompts = load_grab_df_prompts()
        if grab_seq_paths is not None:
            grab_seq_names = [
                grab_seq_path_to_seq_id(seq_path) for seq_path in grab_seq_paths
            ]
            grab_amass_dp_names = []
            for grab_seq_name in grab_seq_names:
                grab_amass_dp_name = get_grab_amass_datapoint_name(
                    grab_seq_name, df_grab_prompts
                )
                if grab_amass_dp_name is None:
                    raise ValueError(
                        f"Grab sequence {grab_seq_name} not found in df_grab_prompts"
                    )
                grab_amass_dp_names.append(grab_amass_dp_name)
            df = df[df["dp_name"].apply(lambda n: n in grab_amass_dp_names)]

        smpl_data_lst = []
        texts_lst = []
        is_grab_lst = []
        dp_names = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Load smpl data"):
            path = row["path"]
            dataset = row["dataset"]
            dp_name = row["dp_name"]

            if lim is not None:
                if len(smpl_data_lst) == lim:
                    break
                # load 1/2 and 1/2
                if dataset != "GRAB" and len(smpl_data_lst) - sum(
                    is_grab_lst
                ) == math.ceil(lim / 2):
                    continue
                if dataset == "GRAB" and sum(is_grab_lst) == math.floor(lim / 2):
                    continue

            if not os.path.exists(path):
                continue
            smpl_data = SmplData.load(path)
            if dataset == "GRAB":
                start, end = get_cut_by_proximity_range(smpl_data)
                smpl_data = smpl_data.cut(start, end)

            if len(smpl_data) < seq_len:
                continue

            if dataset == "GRAB":
                grab_amass_dp_name = get_grab_amass_datapoint_name(
                    dp_name, df_grab_prompts
                )
                if grab_amass_dp_name is None:
                    continue
                texts = [df_grab_prompts.loc[grab_amass_dp_name].Prompt]
            else:
                texts = load_hml3d_texts(dp_name)

            smpl_data_lst.append(smpl_data)
            texts_lst.append(texts)
            is_grab_lst.append(dataset == "GRAB")
            dp_names.append(dp_name)

        print(
            f"#data-points = {len(smpl_data_lst)}; {sum(is_grab_lst)} data-points are GRAB data points"
        )

        self.df_grab_prompts = df_grab_prompts
        self.smpl_data_lst = smpl_data_lst
        self.texts_lst = texts_lst
        self.grab_indices = list(np.where(is_grab_lst)[0])
        self.dp_names = dp_names

    def get_sampler(
        self, p_grab_pair: float = 0.25, out_lim_factor: float = 4
    ) -> MixedSampler:
        return MixedSampler(
            A_indices=list(range(len(self.smpl_data_lst))),
            B_indices=list(self.grab_indices),
            p_B_pair=p_grab_pair,
            epoch_size=len(self.smpl_data_lst),
            output_limit=out_lim_factor
            and int(out_lim_factor * len(self.smpl_data_lst)),
        )

    def random_crop(self, smpl_data: SmplData) -> SmplData:
        start = np.random.randint(len(smpl_data) - self.seq_len + 1)
        end = start + self.seq_len
        smpl_data = smpl_data.cut(start, end)
        return smpl_data

    def __getitem__(
        self, idx: int | Tuple[int, int, int], start_frame: Optional[int] = None
    ):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        if isinstance(idx, tuple):
            idx_body, idx_hands, _ = idx
        else:
            idx_body = idx
            idx_hands = np.random.choice(self.grab_indices)
        assert 0 <= idx_body < len(self.smpl_data_lst)
        assert (
            idx_hands in self.grab_indices
        )  # consider removing to support HML3D training

        if idx_body == idx_hands:
            smpl_data = self.smpl_data_lst[idx_body]
            if start_frame is None:
                smpl_data = self.random_crop(smpl_data)
            else:
                smpl_data = smpl_data.cut(start_frame, start_frame + self.seq_len)

            text = self.texts_lst[idx_body][0]

            text_body = text
            text_hands = text
        else:
            smpl_data_body = self.smpl_data_lst[idx_body]
            smpl_data_hands = self.smpl_data_lst[idx_hands]

            ##############
            # Cut selected motions
            ##############
            smpl_data_body = self.random_crop(smpl_data_body)
            smpl_data_hands = self.random_crop(smpl_data_hands)

            ##############
            # Merge motions features
            ##############
            poses_merged = smpl_data_body.poses.clone()
            poses_merged[:, -30 * 3 :] = smpl_data_hands.poses[:, -30 * 3 :]
            smpl_data_merged = SmplData(
                poses=poses_merged,
                trans=smpl_data_body.trans,
                joints=None,
            )
            with torch.no_grad():
                smplx_out = self.smplx_fk.smpldata_to_smpl_output(
                    smpl_data_merged.to(self.smplx_fk.device),
                )
            smpl_data_merged.joints = smplx_out.joints
            smpl_data = smpl_data_merged

            ##############
            # Merge texts
            ##############
            text_body = np.random.choice(self.texts_lst[idx_body])
            text_hands = self.texts_lst[idx_hands][
                0
            ]  # Each GRAB motion has only one text

        features = self.feature_processor.encode(smpl_data)
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
        text = f"{text_body} [SEP] {text_hands}"

        result = {
            "features": features,
            "text": text,
        }

        if self.condition_input_feature is not None:
            condition_mask_1d = self.feature_processor.get_feature_mask(
                self.condition_input_feature
            )  # (n_features,)
            condition_mask = condition_mask_1d.expand(
                features.shape[0], features.shape[1]
            )  # (seq_len, n_features)
            condition_input = torch.zeros_like(
                features
            )  # it's not really needed to put zeros in the non-conditioned features but anyway...
            condition_input[condition_mask] = features[condition_mask]
            result["condition_mask"] = condition_mask
            result["condition_input"] = condition_input
            assert condition_mask.sum().item() > 0, "No condition input features found"

        if self.use_cache:
            self.cache[idx] = result

        return result


class HoiGrabDataset(SmplRifkeDataset):
    """
    Contains GRAB motions with:
        1. Human motion (Pose + Trans + Joints(redundant))
        2. Object motion (6DoF)
        3. Object-human interaction (contact points)
    """

    def __init__(
        self,
        grab_dataset_path,
        seq_len=60,
        use_cache=True,
        lim=None,
        cut_by_table_proximity=True,
        features_string="hoi_body_hands",
        n_points=1000,
        grab_seq_paths=None,
        grab_split=None,
    ):
        super().__init__()
        self.split = grab_split
        self.grab_dataset_path = grab_dataset_path
        self.seq_len = seq_len
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
            grab_dataset_path, n_points=n_points
        )
        self.features_string = features_string

        if use_cache:
            memory = Memory(os.path.join(PROJECT_DIR, "grab_extended_smpldata_cache"))
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
            features_string=self.features_string,
        )

        df_prompts = load_grab_df_prompts()

        self.data = []
        for ext_smpldata in tqdm(
            ext_smpldata_lst, total=len(ext_smpldata_lst), desc="Loading HoiGrabDataset"
        ):
            seq_path = ext_smpldata["grab_seq_path"]
            smpl_data = ext_smpldata["smpldata"]
            object_name = ext_smpldata["object_name"]
            intent_vec = ext_smpldata["intent_vec"]
            if lim is not None and len(self.data) == lim:
                break
            if cut_by_table_proximity:
                start, end = get_cut_by_proximity_range(smpl_data)
                smpl_data = smpl_data.cut(start, end)
            if len(smpl_data) < self.seq_len:
                continue
            features = self.feature_processor.encode(smpl_data)

            self.data.append(
                {
                    "grab_seq_path": seq_path,
                    "features": features,
                    "smpldata": smpl_data,
                    "text": df_prompts.loc[grab_seq_path_to_seq_id(seq_path)]["Prompt"],
                    "object_name": object_name,
                    "range": (start, end),
                    "intent_vec": intent_vec,
                    "name": grab_seq_path_to_seq_id(seq_path),
                }
            )

    def __len__(self):
        return len(self.data)

    def get_object_pcd(self, obj_name: str):
        pcd_ds_idx = self.obj_pcd_dataset.name2idx[obj_name]
        return self.obj_pcd_dataset[pcd_ds_idx]

    def __getitem__(self, idx, start_frame: Optional[int] = None):
        data = self.data[idx]
        features_seq = data["features"]
        text = data["text"]
        if start_frame is not None:
            start = start_frame
        else:
            start = np.random.randint(len(features_seq) - self.seq_len + 1)
        end = start + self.seq_len

        if self.split != "train":
            start = 0
            end = len(features_seq)

        features = features_seq[start:end]
        contact = data["smpldata"].contact[start:end]
        contact_bits = contact.sum(dim=1) > 0.5
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)

        obj_name = data["object_name"]
        obj_points: PyGData = self.get_object_pcd(obj_name)

        res = {
            "features": features,  # T , 556
            "text": text,
            "obj_points": obj_points,  # 1024,3
            "range": (start, end),
            "contact_bits": contact_bits,
            "intent_vec": data["intent_vec"],
            "smpldata": data["smpldata"],
        }
        if self.split == "test":
            res["intent_vec"] = data["intent_vec"]
            res["name"] = data["name"]
            res["seq_path"] = data["grab_seq_path"]
        return res


def main_hml():
    dataset_path = "/home/dcor/roeyron/trumans_utils/Data_AMASS_smplrifke_inputs"
    batch_size = 32
    n_features = 217
    context_len = 20
    pred_len = 40

    dataset = HML3DSmplRifkeDataset(
        dataset_path, seq_len=context_len + pred_len, lim=100
    )
    mean, std = get_mean_and_std(dataset)
    normalizer = Normalizer(mean, std)
    normalizer.save(dataset_path)
    normalizer.from_dir(dataset_path)
    dataset.set_normalizer(normalizer)
    collate_func = partial(
        collate_smplrifke_mdm, pred_len=pred_len, context_len=context_len
    )
    collate_func = partial(
        collate_smplrifke_mdm, pred_len=pred_len, context_len=context_len
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_func)

    x, cond = next(iter(dataloader))

    y = cond["y"]
    assert x.shape == (batch_size, n_features, 1, pred_len)
    assert y["prefix"].shape == (batch_size, n_features, 1, context_len)
    assert "text" in y
    assert "mask" in y
    print(y["text"])
    print(123)


def main_grab():
    dataset_path = "/home/dcor/roeyron/trumans_utils/Data_GRAB_smplrifke_inputs"
    batch_size = 32
    n_features = 469
    context_len = 20
    pred_len = 40

    dataset = GrabRifkeDataset(dataset_path, seq_len=context_len + pred_len, lim=100)
    mean, std = get_mean_and_std(dataset)
    normalizer = Normalizer(mean, std)
    normalizer.save(dataset_path)
    normalizer.from_dir(dataset_path)
    dataset.set_normalizer(normalizer)
    collate_func = partial(collate_smplrifke_mdm, pred_len=pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_func)

    x, cond = next(iter(dataloader))

    y = cond["y"]
    assert x.shape == (batch_size, n_features, 1, pred_len)
    assert y["prefix"].shape == (batch_size, n_features, 1, context_len)
    assert "text" in y
    assert "mask" in y
    print(y["text"])
    print(123)


def main_hoi_grab():
    dataset_path = GRAB_DATA_PATH
    batch_size = 4
    n_features = 469 + 3 + 6 + 2 * 30  # 3 obj xyz, 6 cont6d, 2*30 contact points
    context_len = 20
    pred_len = 40
    limit = batch_size
    # device = get_least_busy_device()
    dataset = HoiGrabDataset(
        dataset_path, seq_len=context_len + pred_len, use_cache=False, lim=limit
    )
    collate_func = partial(
        collate_smplrifke_mdm, pred_len=pred_len, context_len=context_len
    )
    collate_func = partial(
        collate_smplrifke_mdm, pred_len=pred_len, context_len=context_len
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_func)

    x, cond = next(iter(dataloader))
    y = cond["y"]
    n_features = x.shape[1]
    assert x.shape == (batch_size, n_features, 1, pred_len)
    assert y["prefix"].shape == (batch_size, n_features, 1, context_len)
    assert "text" in y
    assert "mask" in y
    assert "obj_points" in y
    print(y["text"])

    smpl_data_lst = dataset.smpldata_lst[:2]
    smpl_data_lst = [e.to("cpu") for e in smpl_data_lst]
    with open(os.path.join(SRC_DIR, "resources", "smpldata_lst.pkl"), "wb") as f:
        pickle.dump(smpl_data_lst, f)


if __name__ == "__main__":
    main_hoi_grab()
