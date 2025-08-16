from collections import defaultdict
from enum import Enum
import os
import torch
from typing import Any, List, Optional, Dict, Tuple, Union
from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import se3_exp_map, se3_log_map

from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.cphoi.cphoi_dataset import TfmsManager
from hoidini.cphoi.cphoi_utils import FeaturesDecoderWrapper
from hoidini.datasets.grab.grab_object_records import ContactRecord
from hoidini.datasets.grab.grab_utils import get_MANO_SMPLX_vertex_ids
from hoidini.general_utils import SRC_DIR, read_json
from hoidini.object_contact_prediction.cpdm_dataset import ContactPairsSequence
from hoidini.object_contact_prediction.cpdm_dno_conds import Table
from hoidini.objects_fk import ObjectModel
from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info
from hoidini.datasets.smpldata import SmplData, SmplModelsFK, SmplFK
from hoidini.normalizer import Normalizer
from dataclasses import dataclass
from hoidini.datasets.dataset_smplrifke import samples_to_smpldata_lst
from hoidini.amasstools.smplrifke_feats import SMPLFeatureProcessor
from hoidini.smplx.body_models import SMPLXOutput
from hoidini.geometry3d.hands_intersection_loss import HandIntersectionLoss


class DnoLossSource(Enum):
    FROM_NN = "from_nn"  # aka nearest neighbor
    FROM_CONTACT_RECORD = "from_contact_record"
    FROM_CONTACT_PAIRS = "from_contact_pairs"


@dataclass
class FkResults:
    joints: Optional[Tensor] = None  # (B, seq_len, n_joints, 3)
    verts: Optional[Tensor] = None  # (B, seq_len, n_vers, 3)


class DnoCondBase(ABC):
    def __init__(
        self,
        smpl_fk: SmplFK,
        normalizer: Normalizer,
        feature_processor: SMPLFeatureProcessor,
    ):
        self.smpl_fk = smpl_fk
        self.normalizer = normalizer
        self.feature_processor = feature_processor

    @property
    def device(self):
        return self.normalizer.mean.device

    def run_fk(
        self,
        samples: Tensor,
        fk_start: int = None,
        fk_end: int = None,
        tfms_manager: Optional[TfmsManager] = None,
    ) -> Tuple[List[SMPLXOutput], List[SmplData]]:
        tfms_processor_lst = tfms_manager.tfms_processor_lst if tfms_manager else None
        smpldata_lst_full = samples_to_smpldata_lst(
            samples,
            self.normalizer,
            feature_processor=self.feature_processor,
            tfms_processor_lst=tfms_processor_lst,
        )
        if fk_start or fk_end:
            smpldata_lst_cur = [
                smpl_data.cut(fk_start, fk_end) for smpl_data in smpldata_lst_full
            ]
        smplx_output_lst_cur = self.smpl_fk.smpldata_to_smpl_output_batch(
            smpldata_lst_cur
        )

        if any(["R2" in str(f) for f in self.feature_processor.features]):
            # Fill in using FK only for object features
            for smpldata, smplx_output in zip(smpldata_lst_cur, smplx_output_lst_cur):
                smpldata.fill_in_using_fk(smplx_output)

        return smplx_output_lst_cur, smpldata_lst_cur, smpldata_lst_full
        # return FkResults(
        #     joints=torch.stack([so.joints for so in smpl_output_lst]) if do_joints else None,
        #     verts=torch.stack([so.vertices for so in smpl_output_lst]) if do_verts else None
        # )

    @abstractmethod
    def __call__(self, samples: Tensor):
        # fk_result = self.run_fk(samples)
        # joints = fk_result.joints
        # verts = fk_result.verts
        # Abstract method, to be implemented by subclasses
        pass


def se3_geodesic_loss(rotvec1, trans1, rotvec2, trans2):
    """
    Computes SE(3) geodesic loss per batch element.

    Inputs:
        rotvec1, rotvec2: (B, seq, 3) - axis-angle rotation vectors
        trans1, trans2:   (B, seq, 3) - translations

    Returns:
        loss_per_batch: (B,) - average loss over seq for each batch element
    """

    B, seq, _ = rotvec1.shape

    # Flatten to (B * seq, 3)
    rotvec1_flat = rotvec1.reshape(-1, 3)
    rotvec2_flat = rotvec2.reshape(-1, 3)
    trans1_flat = trans1.reshape(-1, 3)
    trans2_flat = trans2.reshape(-1, 3)

    # Compose SE(3) matrices: (B*seq, 4, 4)
    SE3_1 = se3_exp_map(torch.cat([rotvec1_flat, trans1_flat], dim=-1))
    SE3_2 = se3_exp_map(torch.cat([rotvec2_flat, trans2_flat], dim=-1))

    # Compute relative transform
    delta = torch.bmm(torch.linalg.inv(SE3_2), SE3_1)  # (B*seq, 4, 4)
    log_delta = se3_log_map(delta)  # Either (B*seq, 4, 4) or (B*seq, 6)

    # Compute per-frame loss
    if log_delta.ndim == 3:
        frame_losses = torch.norm(log_delta, dim=(1, 2))  # (B*seq,)
    elif log_delta.ndim == 2:
        frame_losses = torch.norm(log_delta, dim=1)  # (B*seq,)
    else:
        raise ValueError(f"Unexpected log_delta shape: {log_delta.shape}")

    # Reshape and average over sequence: (B, seq) -> (B,)
    loss_per_batch = frame_losses.view(B, seq).mean(dim=1)
    return loss_per_batch


class KeepOnTableWhenNotContactLoss(nn.Module):
    def __init__(self, th_contact_force: float):
        super().__init__()
        self.th_contact_force = th_contact_force

    def forward(self, body_verts: Tensor, contact_records: List[ContactRecord]):
        pass


class EnforceContactBitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src_contact: Tensor, tgt_contact: Tensor):
        """
        src_contact: (B, T, n_anchors)
        tgt_contact: (B, T, n_anchors)
        """
        tgt_contact = (tgt_contact > 0.5).float()
        loss = torch.mean(torch.abs(src_contact - tgt_contact), dim=(1, 2))
        return loss


class GravityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, body_verts: Tensor, contact_records: List[ContactRecord]):
        pass


class JointsJitterLoss(nn.Module):
    def __init__(self, fps: float = 20, p: float = 2.0, threshold: float | None = None):
        super().__init__()
        self.fps = fps
        self.p = p
        self.threshold = threshold

    def forward(self, joints: Tensor):
        return rms_jerk_jitter(joints, fps=self.fps, p=self.p, threshold=self.threshold)


def rms_jerk_jitter(
    xyz: Tensor,
    *,
    fps: float = 20,
    p: float = 2.0,
    threshold: float | None = None,
) -> Tensor:
    """
    Computes jerk-based jitter per batch element.

    Args:
        xyz: Tensor of shape (B, T, J, 3)
        fps: Frames per second - scales jerk to physical units
        p: Exponent to use (e.g., 2 = RMS, 4 = punish big values more)
        threshold: Optional - zero out jerk magnitudes below this

    Returns:
        Tensor of shape (B,) - one jitter score per batch element
    """
    jerk = torch.diff(xyz, n=3, dim=1) * (fps**3)  # (B, T-3, J, 3)
    jerk_mag = torch.linalg.vector_norm(jerk, dim=-1)  # (B, T-3, J)

    if threshold is not None:
        jerk_mag = torch.where(
            jerk_mag < threshold, torch.zeros_like(jerk_mag), jerk_mag
        )

    jitter = torch.mean(jerk_mag**p, dim=(1, 2)) ** (1 / p)  # (B,)
    return jitter


class HandsContactLoss(nn.Module):
    """Lp vertex‑distance loss on the contacting hand vertices."""

    def __init__(
        self,
        th_contact_force: float,
        p: int = 4,
        use_anchors: bool = False,
    ):
        super().__init__()
        self.th_contact_force = th_contact_force
        self.mano_smplx_vertex_ids = get_MANO_SMPLX_vertex_ids()
        self.p = p
        self.use_anchors = use_anchors
        self.anchor_inds_R2hands = (
            get_contact_anchors_info()[0] if use_anchors else None
        )

    def forward(
        self,
        src_body_verts: Tensor,  # (B, T, N_body, 3)
        contact_force_lhand: Tensor,  # (B, T, n_anchors)    or (B, T, N_hand)
        contact_force_rhand: Tensor,  # (B, T, n_anchors)    or (B, T, N_hand)
        tgt_lhand_verts: Tensor,  # (B, T, n_anchors, 3) or (B, T, N_hand, 3)
        tgt_rhand_verts: Tensor,  # (B, T, n_anchors, 3) or (B, T, N_hand, 3)
    ) -> Tensor:  # (B,)
        batch_size = src_body_verts.shape[0]
        assert contact_force_lhand.shape == contact_force_rhand.shape
        assert tgt_lhand_verts.shape == tgt_rhand_verts.shape
        assert contact_force_lhand.shape == tgt_lhand_verts.shape[:-1]

        # Calculate loss for each batch element separately
        loss_contact = []
        for b in range(batch_size):

            curr_body_verts = src_body_verts[
                b : b + 1
            ]  # Keep batch dimension: (1, seq, n_verts, 3)

            l_mask = contact_force_lhand[b] > self.th_contact_force
            r_mask = contact_force_rhand[b] > self.th_contact_force

            l_mask = l_mask[None, :, :, None]  # (1, seq, n_verts, 1)
            r_mask = r_mask[None, :, :, None]  # (1, seq, n_verts, 1)

            batch_loss = torch.tensor([0.0], device=src_body_verts.device)
            if l_mask.sum() > 0:
                l_locs_tgt = tgt_lhand_verts[b][None]  # (1, seq, n_verts, 3)
                l_locs = curr_body_verts[
                    :, :, self.mano_smplx_vertex_ids["left_hand"], :
                ]  # (1, seq, n_verts, 3)
                if self.use_anchors:
                    l_locs = l_locs[:, :, self.anchor_inds_R2hands, :]
                batch_loss += torch.mean(
                    torch.abs((l_locs_tgt - l_locs) * l_mask) ** self.p, dim=(1, 2, 3)
                ) ** (1 / self.p)

            if r_mask.sum() > 0:
                r_locs_tgt = tgt_rhand_verts[b][None]
                r_locs = curr_body_verts[
                    :, :, self.mano_smplx_vertex_ids["right_hand"], :
                ]
                if self.use_anchors:
                    r_locs = r_locs[:, :, self.anchor_inds_R2hands, :]
                batch_loss += torch.mean(
                    torch.abs((r_locs_tgt - r_locs) * r_mask) ** self.p, dim=(1, 2, 3)
                ) ** (1 / self.p)

            loss_contact.append(batch_loss)

        return torch.cat(loss_contact)

    # def forward(
    #     self,
    #     src_body_verts: Tensor,       # (B, T, N_body, 3)
    #     contact_force_lhand: Tensor,  # (B, T, n_anchors)    or (B, T, N_hand)
    #     contact_force_rhand: Tensor,  # (B, T, n_anchors)    or (B, T, N_hand)
    #     tgt_lhand_verts: Tensor,      # (B, T, n_anchors, 3) or (B, T, N_hand, 3)
    #     tgt_rhand_verts: Tensor,      # (B, T, n_anchors, 3) or (B, T, N_hand, 3)
    # ) -> Tensor:                      # (B,)
    #     B, T, _, _ = src_body_verts.shape
    #     device = src_body_verts.device

    #     assert contact_force_lhand.shape == contact_force_rhand.shape
    #     assert tgt_lhand_verts.shape == tgt_rhand_verts.shape
    #     assert contact_force_lhand.shape == tgt_lhand_verts.shape[:-1]

    #     l_mask = (contact_force_lhand > self.th_contact_force).float()
    #     r_mask = (contact_force_rhand > self.th_contact_force).float()

    #     l_ids = self.mano_smplx_vertex_ids["left_hand"]
    #     r_ids = self.mano_smplx_vertex_ids["right_hand"]

    #     if self.use_anchors:
    #         # a = self.anchor_inds_R2hands
    #         l_ids = l_ids[self.anchor_inds_R2hands]
    #         r_ids = r_ids[self.anchor_inds_R2hands]

    #     l_pred = src_body_verts[:, :, l_ids, :]
    #     r_pred = src_body_verts[:, :, r_ids, :]

    #     def masked_lp(pred: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
    #         loss_results = torch.tensor(B * [0.0], device=device)
    #         diff_p = (pred - tgt).abs().pow(self.p)
    #         masked = diff_p * mask.unsqueeze(-1)
    #         denom = mask.sum(dim=(1, 2)).clamp_min(1.0) * 3.0
    #         loss = masked.sum(dim=(1, 2, 3)) / denom
    #         loss = loss.pow(1.0 / self.p)
    #         loss_results += loss
    #         return loss_results

    #     l_loss = masked_lp(l_pred, tgt_lhand_verts, l_mask)
    #     r_loss = masked_lp(r_pred, tgt_rhand_verts, r_mask)
    #     return l_loss + r_loss


def get_contact_record_from_smpldata(
    body_verts: Tensor,
    obj_fk: ObjectModel,
    smpldata: SmplData,
    obj_faces: Optional[Tensor],
    nearest_neighbor_per_frame: bool = True,
    given_closest_obj_vert_inds_dict: Optional[Dict[str, Tensor]] = None,
) -> Tuple[ContactRecord, Dict[str, Tensor]]:
    """
    body_verts: (seq, n_verts, 3)
    """
    # Create a new tensor for body_verts to avoid in-place modifications
    body_verts = body_verts.clone()

    # Compute object vertices without in-place operations
    with torch.no_grad():
        obj_verts = obj_fk(
            transl=smpldata.trans_obj, global_orient=smpldata.poses_obj
        ).vertices.clone()

    mano_smplx_vertex_ids = get_MANO_SMPLX_vertex_ids()
    anchor_inds_R2hands, _, _ = get_contact_anchors_info()
    contact_per_hand = smpldata.get_contact_per_hand()
    hand_vert_locs_dict = {}
    closest_obj_vert_inds_dict = {}
    for hand in ["left", "right"]:
        hand_verts_seq = body_verts[:, mano_smplx_vertex_ids[hand + "_hand"], :]
        anchor_verts_seq = hand_verts_seq[:, anchor_inds_R2hands]
        contact = contact_per_hand[hand]
        contact = contact > 0.5

        if given_closest_obj_vert_inds_dict is None:
            # Compute distances and closest vertices without in-place operations
            dists = torch.cdist(
                anchor_verts_seq, obj_verts
            )  # (seq_len, n_anchors, n_obj_verts)
            closest_obj_vert_inds = torch.argmin(dists, dim=2)  # (seq_len, n_anchors)
            if not nearest_neighbor_per_frame:
                dynamic_blocks = get_contiguous_dynamic_blocks(contact)
                for start, end in dynamic_blocks:
                    shape = closest_obj_vert_inds[
                        start:end
                    ].shape  # (segment_len, n_anchors, 3)
                    closest_obj_vert_inds[start:end] = torch.mode(
                        closest_obj_vert_inds[start:end], dim=0
                    ).values.expand(
                        *shape
                    )  # (segment_len, n_anchors, 3)
            closest_obj_vert_inds_dict[hand] = closest_obj_vert_inds
        else:
            closest_obj_vert_inds = given_closest_obj_vert_inds_dict[hand]
        selected_verts = torch.gather(
            obj_verts,
            dim=1,
            index=closest_obj_vert_inds.unsqueeze(-1).expand(-1, -1, 3),
        )  # (seq_len, n_anchors, 3)
        hand_vert_locs_dict[hand] = selected_verts

    contact_record = ContactRecord(
        obj_verts=obj_verts,
        obj_faces=obj_faces,
        lhand_vert_locs=hand_vert_locs_dict["left"],
        rhand_vert_locs=hand_vert_locs_dict["right"],
        lhand_contact_force=contact_per_hand["left"],
        rhand_contact_force=contact_per_hand["right"],
    )
    return contact_record, closest_obj_vert_inds_dict


def get_contiguous_dynamic_blocks(contact_mask: Tensor) -> List[Tuple[int, int]]:
    """
    Get contiguous dynamic blocks
    contact_mask: (seq_len, n_anchors) boolean tensor indicating contact
    """
    contact_mask = contact_mask.any(dim=1)  # (seq_len,)
    dynamic_blocks = []
    start_idx = None
    for i in range(contact_mask.shape[0]):
        if contact_mask[i]:
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            dynamic_blocks.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        dynamic_blocks.append((start_idx, contact_mask.shape[0]))
    return dynamic_blocks


class ObjectRepresentationsConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, smpldata_lst: List[SmplData], p: int = 1):
        """
        contact: (B, n_anchors, seq_len)
        obj_verts: (B, n_verts, 3)
        """
        poses_obj_body = torch.stack(
            [smpldata.poses_obj for smpldata in smpldata_lst]
        )  # (B, seq, n_verts, 3)
        trans_obj_body = torch.stack(
            [smpldata.trans_obj for smpldata in smpldata_lst]
        )  # (B, seq, 3)
        poses_obj_lhand = torch.stack(
            [smpldata.poses_obj_from_lhand for smpldata in smpldata_lst]
        )  # (B, seq, 3)
        trans_obj_lhand = torch.stack(
            [smpldata.trans_obj_from_lhand for smpldata in smpldata_lst]
        )  # (B, seq, 3)
        poses_obj_rhand = torch.stack(
            [smpldata.poses_obj_from_rhand for smpldata in smpldata_lst]
        )  # (B, seq, 3)
        trans_obj_rhand = torch.stack(
            [smpldata.trans_obj_from_rhand for smpldata in smpldata_lst]
        )  # (B, seq, 3)

        loss_b_l = se3_geodesic_loss(
            poses_obj_body, trans_obj_body, poses_obj_lhand, trans_obj_lhand
        )
        loss_b_r = se3_geodesic_loss(
            poses_obj_body, trans_obj_body, poses_obj_rhand, trans_obj_rhand
        )
        loss_l_r = se3_geodesic_loss(
            poses_obj_lhand, trans_obj_lhand, poses_obj_rhand, trans_obj_rhand
        )
        loss = loss_b_l + loss_b_r + loss_l_r
        return loss


class FootSkateLoss(nn.Module):
    def __init__(self, th_contact: float = 5e-2):
        super().__init__()
        self.th_contact = th_contact

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        joints: Tensor of shape (B, seq_len, n_joints, 3)
        Returns a scalar loss penalizing horizontal foot motion when in contact with the ground.
        """
        foot_inds = [10, 11]
        B, T, _, _ = joints.shape
        foot = joints[:, :, foot_inds, :]  # (B, T, 2, 3)
        floor_height = foot[:, :, :, 2].min(dim=(1)).values  # (B, 2)

        foot_z = foot[..., 2] - floor_height.unsqueeze(1).expand(B, T, 2)  # (B, T, 2)
        contact_mask = foot_z < self.th_contact  # (B, T, 2)

        # Compute horizontal foot velocity
        foot_xy = foot[..., [0, 1]]  # (B, T, 2, 2)
        lin_vel = torch.abs(foot_xy[:, 1:] - foot_xy[:, :-1])  # (B, T-1, 2, 2)

        # Create contact mask for t and t+1
        contact_t = contact_mask[:, :-1]
        contact_tp1 = contact_mask[:, 1:]
        valid_contact = contact_t & contact_tp1  # (B, T-1, 2)
        loss = torch.zeros(B, device=joints.device)
        for b in range(B):
            mask = valid_contact[b]
            if mask.sum() > 0:
                loss[b] = lin_vel[b, mask].mean()
        return loss


class HoiAboveTableLoss(nn.Module):
    """
    Penalize penetration of the object below the table.
    Always active, not just for specific frames.
    """

    def __init__(self, table_corner_pts: torch.Tensor):
        """
        table_corner_pts: (bs, n_corners, 3)
        """
        assert table_corner_pts.dim() == 3
        super().__init__()
        self.table = Table(table_corner_pts).to(dist_util.dev())
        smplx_vert_segs = read_json(
            os.path.join(
                SRC_DIR, "skeletons/vertices_segments/smplx_vert_segmentation.json"
            )
        )
        relevant_segs = [
            "rightHand",
            "leftArm",
            "rightArm",
            "leftHandIndex1",
            "rightHandIndex1",
            "leftForeArm",
            "rightForeArm",
            "leftHand",
            "rightHand",
        ]
        self.relevant_vert_ids = []
        for seg in relevant_segs:
            self.relevant_vert_ids.extend(smplx_vert_segs[seg])
        self.relevant_vert_ids = torch.tensor(
            self.relevant_vert_ids, device=dist_util.dev()
        )

    def forward(self, body_verts: torch.Tensor) -> torch.Tensor:
        """
        body_verts : (bs, seq_len, nV, 3)
        returns    : (bs,)  – mean penetration per sample
        """
        relevant_body_verts = body_verts[:, :, self.relevant_vert_ids, :]
        hinge = self.table.loss_below_surface(relevant_body_verts)
        return hinge.mean(dim=(1, 2))


class HoiSideTableLoss(nn.Module):
    def __init__(self, table_corner_locs: torch.Tensor):
        super().__init__()
        assert table_corner_locs.dim() == 3
        self.table = Table(table_corner_locs).to(dist_util.dev())
        smplx_vert_segs = read_json(
            os.path.join(
                SRC_DIR, "skeletons/vertices_segments/smplx_vert_segmentation.json"
            )
        )
        relevant_segs = ["hips", "spine", "spine1", "spine2"]
        self.relevant_vert_ids = []
        for seg in relevant_segs:
            self.relevant_vert_ids.extend(smplx_vert_segs[seg])
        self.relevant_vert_ids = torch.tensor(
            self.relevant_vert_ids, device=dist_util.dev()
        )

    def forward(self, body_verts: torch.Tensor) -> torch.Tensor:
        """
        body_verts : (bs, seq_len, nV, 3)
        """
        relevant_body_verts = body_verts[:, :, self.relevant_vert_ids, :]
        loss = self.table.loss_side_penetration(relevant_body_verts)
        return loss.mean(dim=(1, 2))


# class Enforce6DofLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, src_obj_trans, src_obj_pose, tgt_obj_trans, tgt_obj_pose):
#         """
#         src_obj_trans: (B, seq_len, 3)
#         src_obj_pose: (B, seq_len, 3)
#         tgt_obj_trans: (B, seq_len, 3)
#         tgt_obj_pose: (B, seq_len, 3)
#         """
#         loss_b_l = se3_geodesic_loss(src_obj_pose, src_obj_trans, tgt_obj_pose, tgt_obj_trans)
#         loss_b_r = se3_geodesic_loss(src_obj_pose, src_obj_trans, tgt_obj_pose, tgt_obj_trans)
#         loss_l_r = se3_geodesic_loss(poses_obj_lhand, trans_obj_lhand, poses_obj_rhand, trans_obj_rhand)


# @dataclass
# class LossCoefficients:
#     contact: float = 1.0
#     penetration: float = 1.0
#     object_representations_consistency: float = 1.0
#     static_object: float = 1.0


class Scheduler(ABC):
    def __init__(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0

    @abstractmethod
    def get_value(self) -> float:
        pass


class TurnOnOffScheduler(Scheduler):
    def __init__(
        self,
        start_on_step: Optional[int] = None,
        stop_on_step: Optional[int] = None,
        value: float = True,
    ):
        super().__init__()
        self.start_on_step = start_on_step if start_on_step is not None else -np.inf
        self.stop_on_step = stop_on_step if stop_on_step is not None else np.inf
        self.value = value

    def __repr__(self):
        return f"TurnOnOffScheduler(start_on_step={self.start_on_step}, stop_on_step={self.stop_on_step}, value={self.value})"

    def get_value(self) -> float:
        if self.current_step < self.start_on_step:
            return 0.0 if isinstance(self.value, float) else False
        elif self.current_step > self.stop_on_step:
            return 0.0 if isinstance(self.value, float) else False
        else:
            return self.value


class MultiStepScheduler(Scheduler):
    def __init__(self, steps: List[int], values: List[float]):
        """
        steps: List[int]
        values: List[float]
        """
        super().__init__()
        self.steps = steps
        self.values = values

    def __repr__(self):
        return f"MultiStepScheduler(steps={self.steps}, values={self.values})"

    def get_value(self) -> float:
        for step, value in zip(self.steps, self.values):
            if self.current_step >= step:
                return value
        return self.values[-1]


LOSS_FUNCTIONS_MAP = {
    "HandsContactLoss": HandsContactLoss,
    "HoiAboveTableLoss": HoiAboveTableLoss,
    "HoiSideTableLoss": HoiSideTableLoss,
    "JointsJitterLoss": JointsJitterLoss,
    "HandsIntersectionsLoss": HandIntersectionLoss,
    # "ObjectRepresentationsConsistencyLoss": ObjectRepresentationsConsistencyLoss,
    # "KeepObjectStaticLoss": KeepObjectStaticLoss,
    # "Specific6DoFLoss": Specific6DoFLoss,
}


class DnoConditionOld:
    def __init__(
        self,
        smpl_fk: SmplModelsFK,
        pred_len: int,
        feature_decoder: FeaturesDecoderWrapper,
        loss_coefficients: Dict[str, Union[float, Scheduler]],
        dno_loss_source: DnoLossSource,
        dno_losses: Dict[str, Any] = None,
        obj_verts_lst: Optional[List[Tensor]] = None,  # for predicted pairs mode
        obj_faces_lst: Optional[List[Tensor]] = None,  # for predicted pairs mode
        obj_fk_lst: Optional[List[ObjectModel]] = None,  # for predicted pairs mode
        nearest_neighbor_per_frame: bool = True,  # related to the prediction of contact records
    ):
        self.smpl_fk = smpl_fk
        self.pred_len = pred_len
        self.feature_decoder = feature_decoder
        self.loss_coefficients = loss_coefficients
        self.obj_verts_lst = obj_verts_lst
        self.obj_faces_lst = obj_faces_lst
        self.obj_fk_lst = obj_fk_lst

        self.dno_losses = dno_losses
        self.nearest_neighbor_per_frame = nearest_neighbor_per_frame

        self.dno_loss_source = dno_loss_source
        self.global_prefix_lst: Optional[List[Tensor]] = None
        self.closest_obj_vert_inds_dict_lst: Optional[List[Dict[str, Tensor]]] = None

        self.contact_record_lst_current: Optional[List[ContactRecord]] = (
            None  # contact record mode
        )
        self.contact_pairs_lst_current: Optional[List[ContactPairsSequence]] = (
            None  # contact pairs sequence mode
        )

    def reset(self):
        self.global_prefix_lst = None
        for loss_name in self.loss_coefficients:
            if isinstance(self.loss_coefficients[loss_name], Scheduler):
                self.loss_coefficients[loss_name].reset()

    def set_current_seq(
        self, current_seq_lst: List[ContactRecord | ContactPairsSequence | Any]
    ):
        if isinstance(current_seq_lst[0], ContactRecord):
            self.contact_record_lst_current = current_seq_lst
        elif isinstance(current_seq_lst[0], ContactPairsSequence):
            self.contact_pairs_lst_current = current_seq_lst
        else:
            raise ValueError(f"Invalid sequence type: {type(current_seq_lst)}")

    def set_global_prefix_lst(self, global_prefix_lst: List[Tensor]):
        self.global_prefix_lst = global_prefix_lst

    def get_loss_coef(self, loss_name: str) -> float:
        if isinstance(self.loss_coefficients[loss_name], Scheduler):
            return self.loss_coefficients[loss_name].get_value()
        else:
            return self.loss_coefficients[loss_name]

    def get_loss_dict(self, samples) -> Dict[str, Tensor]:
        bs, _, _, cur_len = samples.shape
        assert cur_len == self.pred_len
        assert samples.dim() == 4
        if self.global_prefix_lst is not None:
            samples_full = torch.cat(self.global_prefix_lst + [samples], dim=3)
        full_len = samples_full.shape[3]
        cur_start_frame = full_len - cur_len

        # decode smpldata from features
        smpldata_lst_full: List[SmplData] = self.feature_decoder.decode(samples_full)
        smpldata_lst_cur = [
            smpldata.cut(cur_start_frame, None) for smpldata in smpldata_lst_full
        ]

        # Run FK on human only on the current sequence + one frame before (for continuity loss)
        smpldata_lst_cur_p1 = [
            smpl_data.cut(cur_start_frame - 1, None) for smpl_data in smpldata_lst_full
        ]  # extra frame
        smplx_output_lst_cur_p1 = self.smpl_fk.smpldata_to_smpl_output_batch(
            smpldata_lst_cur_p1
        )  # extra frame
        body_verts_cur = torch.stack(
            [so.vertices[1:] for so in smplx_output_lst_cur_p1]
        )  # (B, pred_len, n_verts, 3)
        joints_cur_p1 = torch.stack(
            [smpldata.joints for smpldata in smplx_output_lst_cur_p1]
        )  # (B, pred_len, n_joints, 3)

        # # Run FK on object
        # obj_trans_cur = torch.stack([sd.trans_obj for sd in smpldata_lst_cur])  # (batch, pred_len, 3)
        # obj_poses_cur = torch.stack([sd.poses_obj for sd in smpldata_lst_cur])  # (batch, pred_len, 3)
        # obj_verts_full = [self.obj_fk_lst[b](transl=obj_trans_cur[b], global_orient=obj_poses_cur[b]) for b in range(bs)]

        opt_tgts = self.get_optimization_targets(smpldata_lst_cur, body_verts_cur)
        loss_fn_inputs = {
            "smpldata_lst_full": smpldata_lst_full,
            "cur_start_frame": cur_start_frame,
            "contact_force_lhand": opt_tgts["contact_force_lhand"],
            "contact_force_rhand": opt_tgts["contact_force_rhand"],
            "tgt_lhand_verts": opt_tgts["tgt_lhand_verts"],
            "tgt_rhand_verts": opt_tgts["tgt_rhand_verts"],
            "object_verts_lst": opt_tgts["object_verts_lst"],
            "object_faces_lst": opt_tgts["object_faces_lst"],
            # "obj_verts_full": obj_verts_full,
            "src_body_verts": body_verts_cur,
            "body_verts": body_verts_cur,
            "joints_cur_p1": joints_cur_p1,
        }

        LOSS_FUNCTIONS_ARGS_WIRING = {
            "HandsContactLoss": {
                "_class_": HandsContactLoss,
                "src_body_verts": "src_body_verts",
                "contact_force_lhand": "contact_force_lhand",
                "contact_force_rhand": "contact_force_rhand",
                "tgt_lhand_verts": "tgt_lhand_verts",
                "tgt_rhand_verts": "tgt_rhand_verts",
            },
            "HandIntersectionLoss": {
                "_class_": HandIntersectionLoss,
                "body_verts": "body_verts",
                "object_verts": "object_verts_lst",
                "object_faces": "object_faces_lst",
            },
            "HoiAboveTableLoss": {
                "_class_": HoiAboveTableLoss,
                "body_verts": "body_verts",
            },
            "HoiSideTableLoss": {
                "_class_": HoiSideTableLoss,
                "body_verts": "body_verts",
            },
            "JointsJitterLoss": {
                "_class_": JointsJitterLoss,
                "joints": "joints_cur_p1",
            },
            "FootSkateLoss": {
                "_class_": FootSkateLoss,
                "joints": "joints_cur_p1",
            },
        }

        loss_dict = {}
        for loss_name, loss_fn in self.dno_losses.items():
            loss_fn_cls_name = type(loss_fn).__name__
            mapping = LOSS_FUNCTIONS_ARGS_WIRING[loss_fn_cls_name]
            kwargs = {
                k: loss_fn_inputs[v]
                for k, v in mapping.items()
                if not k.startswith("_")
            }
            loss_dict[loss_name] = loss_fn(**kwargs)
        return loss_dict

    def get_loss_dict0(
        self, samples, update_nn_verts: bool = True
    ) -> Dict[str, Tensor]:
        assert samples.dim() == 4
        if self.global_prefix_lst is not None:
            samples = torch.cat(self.global_prefix_lst + [samples], dim=3)
        # return one more frame to enable continuity loss
        _smplx_output_lst, _smpldata_lst = self.run_fk(
            samples,
            fk_start=-self.smpl_fk.sbj_model.batch_size,
            tfms_manager=self.tfms_manager,
        )

        smpldata_lst = [smpldata.cut(1, None) for smpldata in _smpldata_lst]

        bs = len(smpldata_lst)
        body_verts = torch.stack(
            [so.vertices[1:] for so in _smplx_output_lst]
        )  # (B, seq, n_verts, 3)
        opt_tgts = self.get_optimization_targets(
            update_nn_verts, smpldata_lst, bs, body_verts
        )

        loss_dict = {}

        # contact loss
        loss_dict["contact"] = self.hand_contact_loss(
            src_body_verts=body_verts,
            contact_force_lhand=opt_tgts["contact_force_lhand"],
            contact_force_rhand=opt_tgts["contact_force_rhand"],
            tgt_lhand_verts=opt_tgts["tgt_lhand_verts"],
            tgt_rhand_verts=opt_tgts["tgt_rhand_verts"],
        )

        # penetration loss
        loss_dict["penetration"] = self.hand_inters_loss(
            body_verts=body_verts,
            object_verts=opt_tgts["object_verts_lst"],
            object_faces=opt_tgts["object_faces_lst"],
        )

        # enforce 6dof loss
        if (
            "6dof" in self.loss_coefficients
            and self.dno_loss_source == DnoLossSource.FROM_CONTACT_PAIRS
        ):
            enforce_6dof_body_loss = se3_geodesic_loss(
                rotvec1=torch.stack([smpldata.poses_obj for smpldata in smpldata_lst]),
                trans1=torch.stack([smpldata.trans_obj for smpldata in smpldata_lst]),
                rotvec2=opt_tgts["object_poses_lst"],
                trans2=opt_tgts["object_trans_lst"],
            )
            enforce_6dof_lhand_loss = se3_geodesic_loss(
                rotvec1=torch.stack(
                    [smpldata.poses_obj_from_lhand for smpldata in smpldata_lst]
                ),
                trans1=torch.stack(
                    [smpldata.trans_obj_from_lhand for smpldata in smpldata_lst]
                ),
                rotvec2=opt_tgts["object_poses_lst"],
                trans2=opt_tgts["object_trans_lst"],
            )
            enforce_6dof_rhand_loss = se3_geodesic_loss(
                rotvec1=torch.stack(
                    [smpldata.poses_obj_from_rhand for smpldata in smpldata_lst]
                ),
                trans1=torch.stack(
                    [smpldata.trans_obj_from_rhand for smpldata in smpldata_lst]
                ),
                rotvec2=opt_tgts["object_poses_lst"],
                trans2=opt_tgts["object_trans_lst"],
            )
            loss_dict["six_dof"] = (
                enforce_6dof_body_loss
                + enforce_6dof_lhand_loss
                + enforce_6dof_rhand_loss
            )

        # object representations consistency
        if "obj_repr_consistency" in self.loss_coefficients:
            loss_dict["obj_repr_consistency"] = ObjectRepresentationsConsistencyLoss()(
                smpldata_lst
            )

        # enforce contact bits loss
        if "contact_bits" in self.loss_coefficients:
            loss_dict["contact_bits"] = EnforceContactBitsLoss()(
                src_contact=torch.stack(
                    [smpldata.contact for smpldata in smpldata_lst]
                ),
                tgt_contact=opt_tgts["contact_force_lhand"],
            )
        if "jitter" in self.loss_coefficients:
            loss_dict["jitter"] = rms_jerk_jitter(
                xyz=torch.stack(
                    [
                        torch.cat(
                            [smpldata.joints[1:].clone().detach(), smpldata.joints[:1]],
                            dim=0,
                        )
                        for smpldata in _smpldata_lst
                    ]
                ),  # use joints from one previous frame as well
                fps=20,
                p=2,
                threshold=70,
            )

        if "above_table" in self.loss_coefficients:
            loss_dict["above_table"] = self.above_table_loss(
                body_verts=body_verts,
            )
        if "side_table" in self.loss_coefficients:
            loss_dict["side_table"] = self.side_table_loss(
                body_verts=body_verts,
            )

        # if "keep_object_static" in self.loss_coefficients:
        #     loss_dict['keep_object_static'] = self.keep_object_static_loss(smpldata_lst)

        return loss_dict

    def get_optimization_targets(self, smpldata_lst, body_verts):
        bs = len(smpldata_lst)
        if self.dno_loss_source == DnoLossSource.FROM_CONTACT_RECORD:
            # use contact record from data - same for all batch elements
            raise NotImplementedError("Not supported for list for now")
            opt_tgts = {
                "contact_force_lhand": self.contact_record_lst_current.lhand_contact_force.expand(
                    bs, -1, -1
                ),
                "contact_force_rhand": self.contact_record_lst_current.rhand_contact_force.expand(
                    bs, -1, -1
                ),
                "tgt_lhand_verts": self.contact_record_lst_current.lhand_vert_locs.expand(
                    bs, -1, -1, -1
                ),
                "tgt_rhand_verts": self.contact_record_lst_current.rhand_vert_locs.expand(
                    bs, -1, -1, -1
                ),
                "object_verts_lst": [
                    self.contact_record_lst_current.obj_verts for _ in range(bs)
                ],
                "object_faces_lst": [
                    self.contact_record_lst_current.obj_faces for _ in range(bs)
                ],
            }
        elif self.dno_loss_source == DnoLossSource.FROM_NN:
            # create contact record from smpldata - one per batch element
            if self.update_nn_verts:
                obj_vert_inds = None
                self.closest_obj_vert_inds_dict_lst = []
            else:
                obj_vert_inds = self.closest_obj_vert_inds_dict_lst
            contact_records = []
            for b in range(len(smpldata_lst)):
                contact_record, closest_obj_vert_inds_dict = (
                    get_contact_record_from_smpldata(
                        body_verts[b],
                        self.obj_fk_lst[b],
                        smpldata_lst[b],
                        self.obj_faces_lst[b],
                        given_closest_obj_vert_inds_dict=(
                            obj_vert_inds[b] if not self.update_nn_verts else None
                        ),
                        nearest_neighbor_per_frame=self.nearest_neighbor_per_frame,
                    )
                )
                contact_records.append(contact_record)
                if self.update_nn_verts:
                    self.closest_obj_vert_inds_dict_lst.append(
                        closest_obj_vert_inds_dict
                    )

            opt_tgts = {
                "contact_force_lhand": torch.stack(
                    [
                        contact_record.lhand_contact_force
                        for contact_record in contact_records
                    ]
                ),
                "contact_force_rhand": torch.stack(
                    [
                        contact_record.rhand_contact_force
                        for contact_record in contact_records
                    ]
                ),
                "tgt_lhand_verts": torch.stack(
                    [
                        contact_record.lhand_vert_locs
                        for contact_record in contact_records
                    ]
                ),
                "tgt_rhand_verts": torch.stack(
                    [
                        contact_record.rhand_vert_locs
                        for contact_record in contact_records
                    ]
                ),
                "object_verts_lst": [
                    contact_record.obj_verts for contact_record in contact_records
                ],
                "object_faces_lst": [
                    contact_record.obj_faces for contact_record in contact_records
                ],
            }
        elif self.dno_loss_source == DnoLossSource.FROM_CONTACT_PAIRS:
            contact_pairs_lst: List[ContactPairsSequence] = (
                self.contact_pairs_lst_current
            )
            opt_tgts = defaultdict(list)
            for b, cp in enumerate(contact_pairs_lst):
                dists = torch.cdist(
                    cp.local_object_points, self.obj_verts_lst[b]
                )  # (seq_len, n_anchors, n_obj_verts)
                closest_obj_vert_inds = torch.argmin(
                    dists, dim=2
                )  # (seq_len, n_anchors)
                with torch.no_grad():
                    obj_verts_seq = self.obj_fk_lst[b](
                        transl=cp.object_trans, global_orient=cp.object_poses
                    ).vertices.clone()  # (seq_len, n_obj_verts, 3)
                anchor_locs = torch.gather(
                    obj_verts_seq,
                    dim=1,
                    index=closest_obj_vert_inds.unsqueeze(-1).expand(-1, -1, 3),
                )
                n_anchors = cp.local_object_points.shape[1] // 2
                opt_tgts["tgt_lhand_verts"].append(anchor_locs[:, :n_anchors])
                opt_tgts["tgt_rhand_verts"].append(anchor_locs[:, n_anchors:])
                opt_tgts["contact_force_lhand"].append(cp.contacts[:, :n_anchors])
                opt_tgts["contact_force_rhand"].append(cp.contacts[:, n_anchors:])
                opt_tgts["object_verts_lst"].append(obj_verts_seq)
                opt_tgts["object_faces_lst"].append(self.obj_faces_lst[b])
            opt_tgts["tgt_lhand_verts"] = torch.stack(opt_tgts["tgt_lhand_verts"])
            opt_tgts["tgt_rhand_verts"] = torch.stack(opt_tgts["tgt_rhand_verts"])
            opt_tgts["contact_force_lhand"] = torch.stack(
                opt_tgts["contact_force_lhand"]
            )
            opt_tgts["contact_force_rhand"] = torch.stack(
                opt_tgts["contact_force_rhand"]
            )
        else:
            raise ValueError(f"Invalid dno loss source: {self.dno_loss_source}")

        return opt_tgts

    def get_predicted_pairs(self, smpldata_lst: List[SmplData]):
        bs = len(smpldata_lst)
        object_verts = []
        object_faces = []
        contact_force_lhand = []
        contact_force_rhand = []
        tgt_lhand_verts = []
        tgt_rhand_verts = []
        for b in range(bs):
            smpldata = smpldata_lst[b]
            local_object_points = smpldata.local_object_points
            obj_verts = self.obj_verts_lst[b]
            dists = torch.cdist(
                local_object_points, obj_verts
            )  # (seq_len, n_anchors, n_obj_verts)
            closest_obj_vert_inds = torch.argmin(dists, dim=2)  # (seq_len, n_anchors)
            obj_fk = self.obj_fk_lst[b]
            with torch.no_grad():
                obj_verts_seq = obj_fk(
                    transl=smpldata.trans_obj, global_orient=smpldata.poses_obj
                ).vertices.clone()  # (seq_len, n_obj_verts, 3)
            anchor_locs = torch.gather(
                obj_verts_seq,
                dim=1,
                index=closest_obj_vert_inds.unsqueeze(-1).expand(-1, -1, 3),
            )
            n_anchors = anchor_locs.shape[1] // 2
            tgt_lhand_verts.append(anchor_locs[:, :n_anchors])
            tgt_rhand_verts.append(anchor_locs[:, n_anchors:])
            object_verts.append(obj_verts_seq)
            object_faces.append(self.obj_faces_lst[b])
            contact_force_lhand.append(smpldata.contact[:, :n_anchors])
            contact_force_rhand.append(smpldata.contact[:, n_anchors:])

        tgt_lhand_verts = torch.stack(tgt_lhand_verts)
        tgt_rhand_verts = torch.stack(tgt_rhand_verts)
        contact_force_lhand = torch.stack(contact_force_lhand)
        contact_force_rhand = torch.stack(contact_force_rhand)
        return {
            "contact_force_lhand": contact_force_lhand,
            "contact_force_rhand": contact_force_rhand,
            "tgt_lhand_verts": tgt_lhand_verts,
            "tgt_rhand_verts": tgt_rhand_verts,
            "object_verts": object_verts,
            "object_faces": object_faces,
        }

    def __call__(self, samples):
        # update_nn_verts = self.get_loss_coef("update_nn_verts")
        loss_dict = self.get_loss_dict(samples)
        loss = torch.zeros(len(samples), device=samples.device)
        for loss_name, loss_coef in self.loss_coefficients.items():
            if loss_coef is None:
                continue
            loss += self.get_loss_coef(loss_name) * loss_dict[loss_name]
        for loss_coef in self.loss_coefficients.values():
            if isinstance(loss_coef, Scheduler):
                loss_coef.step()
        return loss, loss_dict
