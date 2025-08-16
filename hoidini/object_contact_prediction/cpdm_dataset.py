from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as PyGData
from torch.nn import functional as F

from hoidini.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from hoidini.base_data_structure import SequentialData
from hoidini.datasets.dataset_smplrifke import (
    SamplerWithStartFrame,
    collate_smplrifke_mdm,
    get_mean_and_std,
    load_grab_df_prompts,
)
from hoidini.datasets.grab.grab_object_records import (
    ContactRecord,
    extract_grab_contact_data,
)
from hoidini.datasets.grab.grab_utils import (
    get_all_grab_seq_paths,
    get_grab_split_ids,
    grab_seq_path_to_seq_id,
)
from hoidini.normalizer import Normalizer
from hoidini.object_conditioning.object_pointcloud_dataset import (
    GrabObjPcdDataset,
    get_grab_point_cloud_dataset,
)
from hoidini.resource_paths import GRAB_DATA_PATH
from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info


@dataclass
class CpsMetadata:
    grab_seq_path: str = None
    object_name: str = None
    text: str = None


@dataclass
class ContactPairsSequence(SequentialData):
    local_object_points: Tensor  # (seq_len, 2 * n_anchors, 3) - 3D coordinates of NN object point per anchor in local object space
    contacts: Tensor  # (seq_len, 2 * n_anchors) - contact force per anchor
    object_poses: Tensor  # (seq_len, 3) - Global orientation of the object as euler angles or axis-angle
    object_trans: (
        Tensor  # (seq_len, 3) - Global translation of the object in world space
    )
    metadata: Optional[CpsMetadata] = None

    def interpolate(self, src_fps: int, tgt_fps: int) -> "ContactPairsSequence":
        scale = tgt_fps / src_fps  # float – interpolate can handle non‑integer

        def _time_interp(x: Tensor, scale_factor: float) -> Tensor:
            """Resample a tensor whose *first* dim is time (T, …)."""
            if scale_factor == 1.0:
                return x.clone()

            t, *feat = x.shape  # (T, f1, f2, …)
            x_flat = x.reshape(t, -1)  # (T, C)
            x_flat = x_flat.t().unsqueeze(0)  # (1, C, T) – batch‑chan‑time

            x_up = F.interpolate(
                x_flat,
                scale_factor=scale_factor,
                mode="linear",
                align_corners=True,
            )

            up_t = x_up.shape[-1]  # new time length
            return x_up.squeeze(0).t().reshape(up_t, *feat)

        d: Dict = self.clone().to_dict()
        for name in self.get_sequential_attr_names():
            d[name] = _time_interp(d[name], scale)
        d["metadata"] = deepcopy(d.get("metadata", None))
        return self.__class__(**d)

    def __len__(self):
        return self.local_object_points.shape[0]

    @classmethod
    def get_sequential_attr_names(cls):
        return ["local_object_points", "contacts", "object_poses", "object_trans"]

    @classmethod
    def get_spatial_attr_names(cls):
        return ["object_trans"]


def load_contact_pairs_sequence(
    grab_seq_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    tgt_fps: int = 20,
) -> tuple[ContactPairsSequence, str]:
    """
    Load a contact pair sequence from a grab sequence path.
    Args:
        grab_seq_path: Path to the grab sequence file
        start_frame: Start frame
        end_frame: End frame
    Returns:
        contact_pair_sequence: ContactPairSequence
        object_name: str
    """
    grab_contact_data = extract_grab_contact_data(
        grab_seq_path, start_frame=start_frame, end_frame=end_frame, tgt_fps=tgt_fps
    )
    contact_record = ContactRecord(
        obj_verts=grab_contact_data["obj_verts"],
        obj_faces=grab_contact_data["obj_faces"],
        lhand_vert_locs=grab_contact_data["lhand_vert_locs"],
        rhand_vert_locs=grab_contact_data["rhand_vert_locs"],
        lhand_contact_force=grab_contact_data["lhand_contact_force"],
        rhand_contact_force=grab_contact_data["rhand_contact_force"],
    )
    anchor_inds_R2hands, _, _ = get_contact_anchors_info()
    contact_record.reduce_contact_to_anchors()
    contact_record.lhand_contact_force = 1.0 * (
        contact_record.lhand_contact_force > 0.0
    )
    contact_record.rhand_contact_force = 1.0 * (
        contact_record.rhand_contact_force > 0.0
    )

    # local object points
    local_object_points = []
    for hand in ["left", "right"]:
        hand_vert_locs = (
            contact_record.lhand_vert_locs
            if hand == "left"
            else contact_record.rhand_vert_locs
        )
        dists = torch.cdist(
            hand_vert_locs[:, anchor_inds_R2hands], contact_record.obj_verts
        )  # (seq_len, n_anchors, n_obj_verts)
        nearest_obj_verts_inds = torch.argmin(dists, dim=2)  # (seq_len, n_anchors)
        hand_local_object_points = grab_contact_data["obj_v_template"][
            nearest_obj_verts_inds
        ]
        local_object_points.append(hand_local_object_points)
    local_object_points = torch.cat(local_object_points, dim=1)

    # contacts
    contacts = []
    for hand in ["left", "right"]:
        hand_contact = (
            contact_record.lhand_contact_force
            if hand == "left"
            else contact_record.rhand_contact_force
        )
        hand_contact = hand_contact[:, anchor_inds_R2hands]
        contacts.append(hand_contact)
    contacts = torch.cat(contacts, dim=1)

    if torch.any(contacts > 1e4):
        assert False, f"Found contact force greater than 1e6 at {grab_seq_path}"

    metadata = CpsMetadata(
        grab_seq_path=grab_seq_path,
        object_name=grab_contact_data["object_name"],
    )

    contact_pair_sequence = ContactPairsSequence(
        local_object_points=local_object_points,
        contacts=contacts,
        object_poses=grab_contact_data["obj_poses"],
        object_trans=grab_contact_data["obj_trans"],
        metadata=metadata,
    )

    assert (
        contact_pair_sequence.local_object_points.abs().max() < 100
    ), f"contact_pair_sequence.local_object_points.abs().max() = {contact_pair_sequence.local_object_points.abs().max()}"
    return contact_pair_sequence, grab_contact_data["object_name"]


def axis_angle_to_cont6d(poses: Tensor) -> Tensor:
    """
    (*, seq_len, 3) -> (*, seq_len, 6)
    """
    poses_mat = axis_angle_to_matrix(poses)
    cont6d = matrix_to_rotation_6d(poses_mat)
    return cont6d


def cont6d_to_axis_angle(cont6d: Tensor) -> Tensor:
    """
    (*, seq_len, 6) -> (*, seq_len, 3)
    """
    poses_mat = rotation_6d_to_matrix(cont6d)
    poses = matrix_to_axis_angle(poses_mat)
    return poses


DEFAULT_FEATURE_NAMES = [
    "local_object_points",
    "contact",
    "object_poses_cont6d",
    "object_trans",
]
FEATURE_SIZES = {
    "local_object_points": lambda n_anchors: 2 * 3 * n_anchors,
    "contact": lambda n_anchors: 2 * n_anchors,
    "object_poses_cont6d": 6,
    "object_trans": 3,
    "object_velocity": 3,
}


def _integrate_with_correction(vel, abs_pred, beta):
    """
    vel      : (T, 3)  - raw velocities (t → t+1)
    abs_pred : (T, 3)  - absolute positions
    beta     : float   - 0 → no correction, 1 → trust only abs_pred
    """
    base = abs_pred[0]  # (3,)
    pos = [base]  # list for speed

    for t in range(1, vel.size(0)):
        # position after naive integration
        pos_int = pos[-1] + vel[t - 1]

        # proportional correction toward abs_pred[t]
        corr_vel = beta * (abs_pred[t] - pos_int)

        # corrected velocity
        v_star = vel[t - 1] + corr_vel

        # integrate
        pos.append(pos[-1] + v_star)

    return torch.stack(pos, 0)


class FeatureProcessor:
    def __init__(self, n_anchors: int, feature_names: Optional[List[str] | str] = None):
        """
        Args:
            n_anchors: Number of anchors per hand
        """
        self.n_anchors = n_anchors
        if feature_names is None:
            feature_names = DEFAULT_FEATURE_NAMES
        if isinstance(feature_names, str):
            feature_names = feature_names.split("|")

        self.feature_names = feature_names
        sizes = {}
        for feature_name in self.feature_names:
            size_val = FEATURE_SIZES[feature_name]
            if callable(size_val):
                size_val = size_val(self.n_anchors)
            sizes[feature_name] = size_val

        self.sizes = sizes
        self.n_features = sum(sizes.values())

    def encode_features(self, contact_pair_sequence: ContactPairsSequence) -> Tensor:
        seq_len = len(contact_pair_sequence)

        feature_components = {}
        for name in self.feature_names:
            if name == "local_object_points":
                feature_components[name] = (
                    contact_pair_sequence.local_object_points.reshape(
                        seq_len, self.sizes["local_object_points"]
                    )
                )
            elif name == "contact":
                feature_components[name] = contact_pair_sequence.contacts
            elif name == "object_poses_cont6d":
                feature_components[name] = axis_angle_to_cont6d(
                    contact_pair_sequence.object_poses
                )
            elif name == "object_trans":
                feature_components[name] = contact_pair_sequence.object_trans
            elif name == "object_velocity":
                locs = contact_pair_sequence.object_trans  # (seq_len, 3)
                vel = locs[1:] - locs[:-1]
                vel = torch.cat([vel, vel[-1:]], dim=0)
                feature_components[name] = vel

        features = []
        for name in self.feature_names:
            features.append(feature_components[name])
        return torch.cat(features, dim=1)

    def decode_features(
        self,
        features: torch.Tensor,
        trans_mode: str = "abs",  # "abs" | "vel" | "vel_ema" | "vel_corr"
        ema_alpha: float = 0.2,  # for "vel_ema"
        vel_corr_beta: float = 0.1,  # for "vel_corr"
    ) -> ContactPairsSequence:
        """
        Reconstruct a ContactPairsSequence from the flat feature tensor.

        Parameters
        ----------
        features : Tensor
            Shape (seq_len, n_features).
        trans_mode : {"abs", "vel", "vel_ema", "vel_corr"}, default "abs"
            How to decode object translation:
                • "abs"      - use stored absolute positions.
                • "vel"      - integrate stored velocities only.
                • "vel_ema"  - integrate velocities, then blend with the
                            absolute signal using weight `blend_alpha`.
                • "vel_corr" - integrate velocities, then correct for the
                            absolute signal using weight `beta`.
        ema_alpha : float, default 0.2
            Weight for "vel_ema": 0 → only velocity, 1 → only absolute.
        beta : float, default 0.2
            Weight for "vel_corr": 0 → only velocity, 1 → only absolute.
        """
        seq_len = features.shape[0]

        # -------- split the flat feature tensor --------
        feature_components: dict[str, torch.Tensor] = {}
        start = 0
        for name in self.feature_names:
            size = self.sizes[name]
            feature_components[name] = features[:, start : start + size]
            start += size

        abs_pred = feature_components["object_trans"]  # (T, 3)
        obj_trans = decode_global_trans(
            trans_mode, ema_alpha, vel_corr_beta, feature_components, abs_pred
        )

        # -------- package the sequence --------
        return ContactPairsSequence(
            local_object_points=feature_components["local_object_points"].reshape(
                seq_len, 2 * self.n_anchors, 3
            ),
            contacts=feature_components["contact"],
            object_poses=cont6d_to_axis_angle(
                feature_components["object_poses_cont6d"]
            ),
            object_trans=obj_trans,
        )


def decode_global_trans(
    trans_mode, ema_alpha, vel_corr_beta, feature_components, abs_pred
):
    if trans_mode == "abs":
        obj_trans = abs_pred

    else:
        vel = feature_components["object_velocity"]  # (T, 3)
        base = abs_pred[0]  # (3,)

        # exclusive cumulative sum from velocity
        disps = torch.cat(
            [torch.zeros_like(base).unsqueeze(0), vel[:-1]],
            dim=0,
        )
        vel_int = base + torch.cumsum(disps, dim=0)  # (T, 3)

        if trans_mode == "vel":
            obj_trans = vel_int
        elif trans_mode == "vel_ema":
            obj_trans = (1.0 - ema_alpha) * vel_int + ema_alpha * abs_pred
        elif trans_mode == "vel_corr":
            obj_trans = _integrate_with_correction(vel, abs_pred, beta=vel_corr_beta)
        else:
            raise ValueError(
                f"Unknown trans_mode '{trans_mode}'. "
                "Choose from 'abs', 'vel', or 'vel_ema'."
            )

    return obj_trans


class ContactPairsDataset(Dataset):
    def __init__(
        self,
        grab_dataset_path: str,
        lim: Optional[int] = None,
        seq_len: int = 60,
        n_points: int = 128,
        seed: Optional[int] = None,
        grab_seq_paths: Optional[List[str]] = None,
        pcd_augment_rot_z: bool = False,
        pcd_augment_jitter: bool = False,
        fps: int = 20,
        feature_names: Optional[List[str] | str] = None,
        grab_split: Optional[str] = None,
    ):

        super().__init__()
        self.normalizer: Optional[Normalizer] = None
        self.seq_len = seq_len
        if grab_seq_paths is None:
            grab_seq_paths = get_all_grab_seq_paths()

            if grab_split is not None:
                grab_split_ids = set(get_grab_split_ids(grab_split))
                grab_seq_paths = [
                    seq_path
                    for seq_path in grab_seq_paths
                    if grab_seq_path_to_seq_id(seq_path) in grab_split_ids
                ]

            if lim is not None:
                grab_seq_paths = list(
                    np.random.RandomState(seed).permutation(grab_seq_paths)
                )
        self.obj_pcd_dataset: GrabObjPcdDataset = get_grab_point_cloud_dataset(
            grab_dataset_path,
            n_points=n_points,
            augment_rot_z=pcd_augment_rot_z,
            augment_jitter=pcd_augment_jitter,
            use_cache=True,
        )
        grab_seq_names = [
            grab_seq_path_to_seq_id(seq_path) for seq_path in grab_seq_paths
        ]

        n_anchors = len(get_contact_anchors_info()[0])
        self.df_prompts = load_grab_df_prompts()
        self.feature_processor = FeatureProcessor(
            n_anchors=n_anchors, feature_names=feature_names
        )
        self.grab_seq_paths = []
        self.grab_seq_names = []
        self.contact_pair_sequences = []
        self.features_lst = []
        self.text_lst = []
        self.object_names = []

        for i, grab_seq_path in tqdm(
            enumerate(grab_seq_paths),
            desc="Loading contact pair sequences and features",
            total=len(grab_seq_paths),
        ):
            contact_pair_sequence, object_name = load_contact_pairs_sequence(
                grab_seq_path, tgt_fps=fps
            )
            if len(contact_pair_sequence) < seq_len:
                continue
            if lim is not None and len(self.contact_pair_sequences) == lim:
                break
            self.grab_seq_paths.append(grab_seq_path)
            self.grab_seq_names.append(grab_seq_names[i])
            self.contact_pair_sequences.append(contact_pair_sequence)
            self.features_lst.append(
                self.feature_processor.encode_features(contact_pair_sequence)
            )
            self.text_lst.append(
                self.df_prompts.loc[grab_seq_path_to_seq_id(grab_seq_path)]["Prompt"]
            )
            self.object_names.append(object_name)

        print(
            f"ContactPairsDataset: Ending up using {len(self.grab_seq_paths)}/{len(grab_seq_paths)} grab sequences"
        )

    def set_normalizer(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def __len__(self):
        return len(self.grab_seq_paths)

    def get_object_pcd(self, obj_name: str):
        pcd_ds_idx = self.obj_pcd_dataset.name2idx[obj_name]
        return self.obj_pcd_dataset[pcd_ds_idx]

    @property
    def n_feats(self):
        return self.feature_processor.n_features

    def __getitem__(self, idx: int | Tuple[int, int]):
        if isinstance(idx, tuple):
            idx, start_frame = idx
        else:
            start_frame = None
        features_seq = self.features_lst[idx]
        text = self.text_lst[idx]
        if start_frame is None:
            start_frame = np.random.randint(len(features_seq) - self.seq_len + 1)
        end_frame = start_frame + self.seq_len
        features = features_seq[start_frame:end_frame]
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
        obj_name = self.object_names[idx]
        obj_points: PyGData = self.get_object_pcd(obj_name)

        assert (
            features.abs().max() < 150
        ), f"features.abs().max() = {features.abs().max()}; features = {features}"
        assert not torch.isnan(features).any()

        return {
            "features": features,
            "text": text,
            "obj_points": obj_points,
            "metadata": {
                "object_name": obj_name,
                "grab_seq_path": self.grab_seq_paths[idx],
                "range": (start_frame, end_frame),
            },
        }


def get_contact_pairs_dataloader(
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
    use_sampler: bool = False,
    grab_split: Optional[str] = "train",
    pred_len_dataset: Optional[int] = None,
    is_enforce_motion_length: bool = False,
):
    if pred_len_dataset is None:
        pred_len_dataset = pred_len
    total_seq_len_ds = context_len + pred_len_dataset
    total_seq_len_model = context_len + pred_len

    dataset = ContactPairsDataset(
        grab_dataset_path,
        seq_len=total_seq_len_ds,
        lim=lim,
        n_points=n_points,
        seed=seed,
        grab_seq_paths=grab_seq_paths,
        pcd_augment_rot_z=pcd_augment_rot_z,
        pcd_augment_jitter=pcd_augment_jitter,
        fps=fps,
        feature_names=feature_names,
        grab_split=grab_split,
    )
    if use_normalizer:
        if is_training:
            mean, std = get_mean_and_std(dataset, sample_size=2000)
            normalizer = Normalizer(mean, std, eps=1e-5)
            normalizer.save(experiment_dir)
        else:
            normalizer = Normalizer.from_dir(experiment_dir)
        dataset.set_normalizer(normalizer)
    else:
        normalizer = None

    if use_sampler:
        sampler = SamplerWithStartFrame(epoch_size=len(dataset))
    else:
        sampler = None

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
        sampler=sampler,
    )
    return dataloader


def main():
    lim = 10
    grab_dataset_path = GRAB_DATA_PATH
    dataset = ContactPairsDataset(grab_dataset_path, lim=lim)
    print(len(dataset))
    contact_pair_sequence = dataset.contact_pair_sequences[0]
    features = dataset.feature_processor.encode_features(contact_pair_sequence)
    contact_pair_sequence_decoded = dataset.feature_processor.decode_features(features)
    assert torch.allclose(
        contact_pair_sequence.local_object_points,
        contact_pair_sequence_decoded.local_object_points,
    )
    assert torch.allclose(
        contact_pair_sequence.contacts, contact_pair_sequence_decoded.contacts
    )
    assert torch.allclose(
        contact_pair_sequence.object_poses,
        contact_pair_sequence_decoded.object_poses,
        atol=3e-4,
    )
    assert torch.allclose(
        contact_pair_sequence.object_trans, contact_pair_sequence_decoded.object_trans
    )


def main2():
    grab_dataset_path = GRAB_DATA_PATH
    dataloader = get_contact_pairs_dataloader(
        grab_dataset_path,
        batch_size=16,
        total_seq_len=60,
        experiment_dir="/home/dcor/roeyron/trumans_utils/src/EXPERIMENTS/cpdm_debug",
        is_training=True,
        lim=16,
    )
    for motion, cond in dataloader:
        print(motion.shape)
        # assert torch.all(cond['y']['prefix'].abs().max() < 1e4)
        # assert torch.all(motion.abs().max() < 1e4)
        print(123)
        # break

    arr = (
        cond["y"]["prefix"].detach().cpu().numpy()
    )  # (batch_size, n_features, 1, pred_len)
    i1, i2, i3, i4 = np.where(arr > 1e4)
    print(i1, i2, i3, i4)
    print(123)
    # dataset: ContactPairsDataset = dataloader.dataset
    # feature_processor = dataset.feature_processor
    # # feature_processor.decode()


def test_interpolate():
    src_fps = 10
    tgt_fps = 20
    cps, _ = load_contact_pairs_sequence(
        "/home/dcor/roeyron/trumans_utils/DATASETS/DATA_GRAB_RETARGETED/s2/teapot_lift.npz",
        tgt_fps=src_fps,
    )
    cps_interpolated = cps.interpolate(src_fps=src_fps, tgt_fps=tgt_fps)
    cps_dict = cps.to_dict()
    cps_interpolated_dict = cps_interpolated.to_dict()
    for attr in ContactPairsSequence.get_sequential_attr_names():
        print(attr, cps_dict[attr].shape, cps_interpolated_dict[attr].shape)


if __name__ == "__main__":
    # main()
    test_interpolate()
