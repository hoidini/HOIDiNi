from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from hoidini.amasstools.geometry import axis_angle_to_matrix
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.datasets.grab.grab_utils import get_table_params, load_mesh
from hoidini.datasets.smpldata import SmplData
from hoidini.general_utils import reshape_mdm_features_to_standard_format
from hoidini.normalizer import Normalizer
from hoidini.object_contact_prediction.cpdm_dataset import (
    ContactPairsSequence,
    FeatureProcessor,
)
from hoidini.smplx.lbs import batch_rodrigues


class DnoLoss(nn.Module):
    @abstractmethod
    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,
        poses: torch.Tensor,
        obj_verts: torch.Tensor,
    ) -> torch.Tensor:
        """
        cur_start_frame: int - start frame of the current autoregressive step
        trans: (bs, seq_len, 3) - the entire sequence of translations including previous autoregressive steps
        poses: (bs, seq_len, 3) - the entire sequence of poses including previous autoregressive steps
        Return loss (bs, )
        """
        pass


class Table(nn.Module):
    def __init__(self, corner_pts: torch.Tensor, eps: float = 1e-8):
        """
        corner_pts: (B, 4, 3) — four CCW corners per table.
        """
        super().__init__()
        if corner_pts.ndim != 3 or corner_pts.shape[1:] != (4, 3):
            raise ValueError("corner_pts must have shape (B, 4, 3)")
        self.eps = eps
        self.register_buffer("corner_pts", corner_pts.float())

        p1, p2, p3, p4 = corner_pts.unbind(1)  # each is (B,3)
        n = torch.cross(p2 - p1, p3 - p1)  # (B,3)
        assert torch.all(n[:, 2] > 0)
        n = n / (n.norm(dim=-1, keepdim=True) + eps)  # unit normal
        self.register_buffer("n", n)

        edges = torch.stack([p2 - p1, p3 - p2, p4 - p3, p1 - p4], dim=1)  # (B,4,3)
        in_norm = torch.cross(n.unsqueeze(1), edges)  # (B,4,3)
        in_norm = in_norm / (edges.norm(dim=-1, keepdim=True) + eps)
        self.register_buffer("in_norm", in_norm)

    def distance_to_surface(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, T, V, 3)
        returns signed distance d of shape (B, T, V)
        """
        if xyz.ndim != 4 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape (B, T, V, 3)")
        B = xyz.shape[0]
        p1 = self.corner_pts[:, 0].view(B, 1, 1, 3)
        n = self.n.view(B, 1, 1, 3)
        return ((xyz - p1) * n).sum(-1)

    def _inside_footprint(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: (B, T, V, 3) — points projected onto table plane
        returns bool mask (B, T, V) indicating “inside polygon”
        """
        B, T, V, _ = proj.shape
        inside = torch.ones(B, T, V, dtype=torch.bool, device=proj.device)
        for i in range(4):
            v = self.corner_pts[:, i].view(B, 1, 1, 3)
            n_edge = self.in_norm[:, i].view(B, 1, 1, 3)
            inside &= ((proj - v) * n_edge).sum(-1) >= -self.eps
        return inside

    def loss_side_penetration(
        self,
        xyz: torch.Tensor,
        surf_margin: float = 2e-2,
        edge_margin: float = 2e-2,
    ) -> torch.Tensor:
        """
        Penalise points that are inside the footprint, near the tabletop
        (|d| < surf_margin), but too close to the edges in the plane.
        Returns (B,T,V) squared-hinge loss.
        """
        if xyz.ndim != 4 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape (B, T, V, 3)")

        d = self.distance_to_surface(xyz)  # (B,T,V)
        close = d.abs() < surf_margin  # narrow vertical band
        if not close.any():
            return torch.zeros_like(d)

        B = d.shape[0]
        n = self.n.view(B, 1, 1, 3)
        proj = xyz - d.unsqueeze(-1) * n  # project onto plane
        inside = self._inside_footprint(proj)  # (B,T,V)

        # signed in-plane distance to each edge (positive == inside)
        dots = [
            (
                (proj - self.corner_pts[:, i].view(B, 1, 1, 3))
                * self.in_norm[:, i].view(B, 1, 1, 3)
            ).sum(-1)
            for i in range(4)
        ]  # list of (B,T,V)
        edge_clear = torch.stack(dots, -1).min(-1).values  # nearest-edge clearance

        pen = torch.relu(edge_margin - edge_clear)  # hinge
        return pen.pow(2) * (close & inside).to(pen.dtype)

    def loss_below_surface(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, T, V, 3)
        returns squared-penetration loss of shape (B, T, V),
        only where points go below the tabletop polygon.
        """
        d = self.distance_to_surface(xyz)  # (B,T,V)
        B = d.shape[0]
        n = self.n.view(B, 1, 1, 3)
        proj = xyz - d.unsqueeze(-1) * n  # project onto plane
        inside = self._inside_footprint(proj)  # (B,T,V)
        pen = torch.relu(-d)  # how far below
        return pen * inside.to(pen.dtype)

    def loss_outside_polygon(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (B, T, V, 3)
        returns hinge penalty for points outside the table polygon (B, T, V).
        """
        d = self.distance_to_surface(xyz)
        B = d.shape[0]
        n = self.n.view(B, 1, 1, 3)
        proj = xyz - d.unsqueeze(-1) * n
        penalty = torch.zeros_like(d)
        for i in range(4):
            v = self.corner_pts[:, i].view(B, 1, 1, 3)
            n_edge = self.in_norm[:, i].view(B, 1, 1, 3)
            dot = ((proj - v) * n_edge).sum(-1)
            penalty += torch.relu(-dot)
        return penalty


def zero_movement_loss(seq: torch.Tensor):
    """
    seq: (bs, seq_len, 3)
    return (bs, )
    """
    # Calculate distance from each element to the first element
    seq_len = seq.shape[1]
    middle = seq_len // 2
    first_element = seq[:, [middle], :]  # (bs, 1, 3)
    diff = seq - first_element  # (bs, seq_len, 3)
    sq_dist = diff.pow(2).sum(dim=-1)  # (bs, seq_len)
    loss_per_b = sq_dist.mean(dim=-1)  # (bs,)
    return loss_per_b


class BulbUpLoss(DnoLoss):
    def __init__(
        self,
        target_frames: List[torch.Tensor],
    ):
        """
        tgt_transl: (bs, 3)
        target_frames: list of length bs, each a 1-D tensor of frame indices
        enforced_angles: {axis_index → fixed angle in radians}, must not include free_axis
        free_axis: which Euler axis is unconstrained (0,1,2)
        """
        super().__init__()
        dev = dist_util.dev()  # (bs,3)
        self.target_frames = [tf.to(dev) for tf in target_frames]

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,  # (bs, seq_len, 3)
        poses: torch.Tensor,  # (bs, seq_len, ?)
        obj_verts: torch.Tensor,  # kept for compatibility, not used here
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = trans.shape
        frames = torch.arange(seq_len, device=trans.device)

        loss = torch.zeros(bs, device=trans.device)
        for b in range(bs):
            mask = (frames >= cur_start_frame) & torch.isin(
                frames, self.target_frames[b]
            )
            if not mask.any():
                continue
            trans = trans[b, mask]  # (n_mask,3)
            poses = poses[b, mask]  # (n_mask,3)

            loss[b] = self.compute_loss(poses)

        return loss

    def compute_loss(self, axis_angles):
        mat = axis_angle_to_matrix(axis_angles)
        v0 = torch.tensor([-1, 0, 0], dtype=torch.float32).to(mat.device)
        v_tgt = torch.tensor([0, 0, 1], dtype=torch.float32).to(mat.device)
        v_pred = mat.transpose(1, 2) @ v0
        loss = (v_pred - v_tgt).pow(2).mean()
        return loss


class EnforceAnyContactLoss(DnoLoss):
    def __init__(self, target_frames: list[torch.Tensor]):
        """
        target_frames: list with length = bs, each element is a 1-D integer tensor with length = seq_len
        """
        super().__init__()
        assert len(target_frames.shape) == 2

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,  # (bs, seq_len, 3)
        poses: torch.Tensor,  # (bs, seq_len, ?)
        obj_verts: torch.Tensor,  # kept for compatibility, not used here
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = trans.shape
        frames = torch.arange(seq_len, device=trans.device)

        loss = torch.zeros(bs, device=trans.device)
        for b in range(bs):
            mask = (frames >= cur_start_frame) & torch.isin(
                frames, self.target_frames[b]
            )
            if not mask.any():
                continue
            contact = smpldata_lst[b][mask].contact  # (seq_len, n_anchors)
            loss[b] += torch.rule(
                -contact.sum(dim=1) + 0.5
            ).mean()  # at each frame, any anchor must be above 0.5

        return loss


def axis_angle_distance(
    axis_angles1: torch.Tensor, axis_angles2: torch.Tensor
) -> torch.Tensor:
    M1 = axis_angle_to_matrix(axis_angles1)
    M2 = axis_angle_to_matrix(axis_angles2)
    R_rel = M1.transpose(-1, -2) @ M2
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(
        cos_theta, -1.0 + 1e-6, 1.0 - 1e-6
    )  # prevent gradient explosion at ±1
    return torch.acos(cos_theta)


class Specific6DoFLoss(DnoLoss):
    def __init__(
        self,
        tgt_transl: torch.Tensor,
        tgt_poses: torch.Tensor,
        target_frames: list[torch.Tensor],
        location_only: bool = False,
    ):
        """
        tgt_transl: (bs, 3)
        tgt_poses: (bs, 3)
        target_frames: list with length = bs, each element is a 1-D integer tensor with length = seq_len
        """
        super().__init__()
        assert tgt_transl.shape == tgt_poses.shape
        assert len(tgt_transl.shape) == 2
        self.tgt_transl = tgt_transl.to(dist_util.dev())
        self.tgt_poses = tgt_poses.to(dist_util.dev())
        self.target_frames = [tf.to(dist_util.dev()) for tf in target_frames]
        self.location_only = location_only

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,  # (bs, seq_len, 3)
        poses: torch.Tensor,  # (bs, seq_len, ?)
        obj_verts: torch.Tensor,  # kept for compatibility, not used here
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = trans.shape
        frames = torch.arange(seq_len, device=trans.device)

        loss = torch.zeros(bs, device=trans.device)
        for b in range(bs):
            mask = (frames >= cur_start_frame) & torch.isin(
                frames, self.target_frames[b]
            )
            if not mask.any():
                continue

            loss_t = (trans[b, mask] - self.tgt_transl[b]).pow(2).mean()
            # loss_p = (poses[b, mask] - self.tgt_poses[b]).pow(2).mean()
            loss_p = axis_angle_distance(poses[b, mask], self.tgt_poses[b]).mean()
            if self.location_only:
                loss[b] = loss_t
            else:
                loss[b] = loss_t + loss_p

        return loss


class TopVertDistLoss(DnoLoss):
    def __init__(self, tgt_transl: torch.Tensor, target_frames: list[torch.Tensor]):
        """
        tgt_transl: (bs, 3)
        target_frames: list with length = bs, each element is a 1-D integer tensor with length = seq_len
        """
        super().__init__()
        assert len(tgt_transl.shape) == 2
        self.tgt_transl = tgt_transl.to(dist_util.dev())
        self.target_frames = [tf.to(dist_util.dev()) for tf in target_frames]

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,  # (bs, seq_len, 3)
        poses: torch.Tensor,  # (bs, seq_len, ?)
        obj_verts: torch.Tensor,  # (bs, seq_len, nV, 3)
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = trans.shape
        frames = torch.arange(seq_len, device=trans.device)

        loss = torch.zeros(bs, device=trans.device)
        for b in range(bs):
            mask = (frames >= cur_start_frame) & torch.isin(
                frames, self.target_frames[b]
            )
            if not mask.any():
                continue
            top_vert_ind = obj_verts[b, mask, :, 2].max(dim=1).indices
            top_vert = obj_verts[b, mask, top_vert_ind]
            loss[b] = (top_vert - self.tgt_transl[b]).pow(2).mean()

        return loss


class KeepStaticNoContactLoss(DnoLoss):
    def __init__(self, target_frames: list[torch.Tensor]):
        super().__init__()
        self.target_frames = [tf.to(dist_util.dev()) for tf in target_frames]
        self.optim_step = 0

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,  # (bs, seq_len, 3)
        poses: torch.Tensor,  # (bs, seq_len, ?)
        obj_verts: torch.Tensor,  # kept for compatibility, not used here
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        self.optim_step += 1
        bs, seq_len = trans.shape[:2]
        frames = torch.arange(seq_len, device=trans.device)

        loss = torch.zeros(bs, device=trans.device)
        for b in range(bs):
            mask = (frames >= cur_start_frame) & torch.isin(
                frames, self.target_frames[b]
            )
            if not mask.any():
                continue
            loss_t = (trans[b, mask] - trans[b, self.src_frame]).pow(2).mean()
            loss_p = (poses[b, mask] - poses[b, self.src_frame]).pow(2).mean()
            loss[b] = loss_t + loss_p

        return loss


class Similar6DoFLoss(DnoLoss):
    def __init__(self, src_frame: int, target_frames: list[torch.Tensor]):
        """
        src_frame      : int
        target_frames  : list (len = batch size) of 1-D integer tensors,
                         each holding the desired frame indices for that sample
        """
        super().__init__()
        self.src_frame = src_frame
        self.target_frames = [tf.to(dist_util.dev()) for tf in target_frames]

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,  # (bs, seq_len, 3)
        poses: torch.Tensor,  # (bs, seq_len, ?)
        obj_verts: torch.Tensor,  # kept for compatibility, not used here
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        bs, seq_len = trans.shape[:2]
        frames = torch.arange(seq_len, device=trans.device)

        loss = torch.zeros(bs, device=trans.device)
        for b in range(bs):
            mask = (frames >= cur_start_frame) & torch.isin(
                frames, self.target_frames[b]
            )
            if not mask.any():
                continue
            loss_t = (trans[b, mask] - trans[b, self.src_frame]).pow(2).mean()
            # loss_p = (poses[b, mask] - poses[b, self.src_frame]).pow(2).mean()
            loss_p = axis_angle_distance(
                poses[b, mask], poses[b, self.src_frame]
            ).mean()
            loss[b] = loss_t + loss_p

        return loss


class ReachTablesLoss(DnoLoss):
    def __init__(
        self,
        table_corners: torch.Tensor,
        target_frames: torch.Tensor,
        k_lowest: int = 5,
        w_contact: float = 0.4,
        w_penetr: float = 1.0,
        w_outside: float = 0.4,
        w_smooth: float = 0.1,
        w_axis: float = 0.3,
    ):
        super().__init__()
        self.table = Table(table_corners).to(dist_util.dev())
        self.target_frames = target_frames.to(dist_util.dev())
        self.k_lowest = k_lowest
        self.w_contact = w_contact
        self.w_penetr = w_penetr
        self.w_outside = w_outside
        self.w_smooth = w_smooth
        self.w_axis = w_axis
        self.local_axes = torch.tensor(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=torch.float32,
        ).to(dist_util.dev())

        # self.register_buffer(
        self.local_axes = torch.tensor(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=torch.float32,
        ).to(dist_util.dev())
        # )

    def _closest_axis_align_loss(self, poses):
        bs, F = poses.shape[:2]
        R = axis_angle_to_matrix(poses.reshape(-1, 3))  # (bs*F,3,3)

        world_axes = (
            torch.matmul(R, self.local_axes.t()).permute(0, 2, 1).reshape(bs, F, 6, 3)
        )  # (bs,F,6,3)

        n = self.table.n.to(world_axes)  # (3,)
        cos = (world_axes * n).sum(-1).abs()  # |cosθ| (bs,F,6)
        best = cos.max(dim=-1).values  # (bs,F)
        return (1.0 - best).mean(dim=1)  # (bs,)

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,
        poses: torch.Tensor,
        obj_verts: torch.Tensor,
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        frames = torch.arange(cur_start_frame + trans.shape[1], device=trans.device)
        mask = torch.isin(frames, self.target_frames)
        if not mask.any():
            return torch.zeros(trans.size(0), device=trans.device)

        verts = obj_verts[:, mask]  # (bs,F,V,3)
        trans = trans[:, mask]
        poses = poses[:, mask]

        signed = self.table.distance_to_surface(verts)  # (bs,F,V)

        k = min(self.k_lowest, signed.size(2))
        lowest_k = torch.topk(signed, k, dim=2, largest=False).values  # (bs,F,k)
        contact_loss = lowest_k.abs().mean(dim=(1, 2))  # (bs,)
        penetration_loss = self.table.loss_below_surface(verts).mean(dim=(1, 2))
        outside_loss = self.table.loss_outside_polygon(verts).mean(dim=(1, 2))
        smooth_loss = zero_movement_loss(trans) + zero_movement_loss(poses)
        axis_align_loss = self._closest_axis_align_loss(poses)

        loss = (
            self.w_contact * contact_loss
            + self.w_penetr * penetration_loss
            + self.w_outside * outside_loss
            + self.w_smooth * smooth_loss
            + self.w_axis * axis_align_loss
        )
        return loss


class AboveTableLoss(DnoLoss):
    """
    Penalize penetration of the object below the table.
    Always active, not just for specific frames.
    """

    def __init__(self, table_corner_pts: torch.Tensor):
        """
        table_corner_pts: (bs, 4, 3)
        """
        super().__init__()
        assert table_corner_pts.ndim == 3
        assert table_corner_pts.shape[1:] == (4, 3)
        self.table = Table(table_corner_pts)

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,
        poses: torch.Tensor,
        obj_verts: torch.Tensor,
        smpldata_lst: Optional[List[SmplData]] = None,
    ) -> torch.Tensor:
        """
        obj_verts : (bs, seq_len, nV, 3)
        returns    : (bs,)  – mean penetration per sample
        """
        frames = torch.arange(trans.shape[1], device=trans.device)
        mask = frames >= cur_start_frame
        obj_verts = obj_verts[:, mask]
        # no frame-masking here; we want *all* frames in the current AR step
        hinge = self.table.loss_below_surface(obj_verts)  # (bs,T,V)
        # hinge = (hinge + 1e-6) ** (0.5) + hinge + hinge ** 2 + hinge ** 4
        hinge = hinge + torch.exp(hinge) - 1.0
        return hinge.mean(dim=(1, 2))  # average over time & vert


def transform_object_points(
    global_orient: torch.Tensor, transl: torch.Tensor, v_template: torch.Tensor
) -> torch.Tensor:
    """
    Rigid-body transform of template vertices over a sequence.

    Args:
        global_orient (torch.Tensor): (B, T, 3) axis-angle rotations (Rodrigues).
        transl        (torch.Tensor): (B, T, 3) object translations.
        v_template    (torch.Tensor): (B, V, 3) rest-pose vertices.

    Returns:
        torch.Tensor: (B, T, V, 3) vertices after applying rotation & translation.
    """
    B, T, _ = global_orient.shape
    _, V, _ = v_template.shape

    # (B*T, 3, 3) rotation matrices
    rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view(B * T, 3, 3)
    # Expand template over time → (B*T, V, 3)
    v_template_seq = v_template.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, V, 3)
    # Apply rigid transform
    verts = torch.matmul(v_template_seq, rot_mats)  # rotate
    verts += transl.view(B * T, 1, 3)  # translate
    return verts.view(B, T, V, 3)


class BatchedObjectModel:
    """
    Work on batches of objects
    """

    def __init__(self, v_template: torch.Tensor):
        """
        v_template: (B, V, 3)
        """
        assert v_template.ndim == 3
        self.v_template = v_template.to(dist_util.dev())

    def __call__(self, poses: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        return transform_object_points(poses, trans, self.v_template)


class CpdmDNOCond:
    def __init__(
        self,
        normalizer: Normalizer,
        feature_processor: FeatureProcessor,
        dno_losses: Dict[str, Tuple[float, DnoLoss]],
        obj_v_template_batch: np.ndarray,
    ):
        self.normalizer = normalizer
        self.feature_processor = feature_processor
        self.global_prefix_lst = None
        self.dno_losses = dno_losses
        self.obj_model = BatchedObjectModel(obj_v_template_batch)

    def __call__(self, samples: torch.Tensor):
        cur_len = samples.shape[3]
        if self.global_prefix_lst is not None:
            samples_full = torch.cat(self.global_prefix_lst + [samples], dim=3)
            for prefix in self.global_prefix_lst:
                assert prefix.detach().is_leaf, "Prefix tensor must be detached"
        else:
            samples_full = samples
        full_len = samples_full.shape[3]
        cur_start_frame = full_len - cur_len
        samples_full = reshape_mdm_features_to_standard_format(samples_full)
        samples_full = [self.normalizer.denormalize(f) for f in samples_full]
        contact_pairs: List[ContactPairsSequence] = [
            self.feature_processor.decode(f, trans_mode="vel_corr")
            for f in samples_full
        ]
        poses_full = torch.stack(
            [cp.object_poses for cp in contact_pairs]
        )  # (batch, seq_len, 3)
        trans_full = torch.stack(
            [cp.object_trans for cp in contact_pairs]
        )  # (batch, seq_len, 3)
        obj_verts_full = self.obj_model(poses_full, trans_full)

        loss = torch.zeros(len(samples), device=samples.device)
        loss_dict = {}
        for k, (w, loss_fn) in self.dno_losses.items():
            cur_loss = loss_fn(cur_start_frame, trans_full, poses_full, obj_verts_full)
            loss += w * cur_loss
            loss_dict[k] = cur_loss.clone().detach()
        return loss, loss_dict

    def reset(self):
        self.global_prefix_lst = None

    def set_global_prefix_lst(self, global_prefix_lst: List[torch.Tensor]):
        self.global_prefix_lst = global_prefix_lst


class RandomOffsetTableLoss(DnoLoss):
    """
    Push the object to a random location on the table while preserving
    its initial orientation.  Offset is sampled once per sequence.
    """

    def __init__(
        self,
        table_corners: torch.Tensor,
        target_frames: torch.Tensor,
        k_lowest: int = 100,
        offset_std: float = 0.08,
        w_offset: float = 1.0,
        w_contact: float = 0.4,
        w_penetr: float = 1.0,
        w_outside: float = 0.4,
        w_smooth: float = 0.1,
    ):
        super().__init__()
        self.table = Table(table_corners).to(dist_util.dev())
        self.target_frames = target_frames.to(dist_util.dev())
        self.k_lowest = k_lowest
        self.offset_std = offset_std
        self.w_offset = w_offset
        self.w_contact = w_contact
        self.w_penetr = w_penetr
        self.w_outside = w_outside
        self.w_smooth = w_smooth
        self._ref_transl = None  # set on first forward
        self._rand_offset = None  # idem

        # in-plane orthonormal basis (t1, t2)
        t1 = self.table.p2 - self.table.p1
        t1 = t1 / t1.norm()
        t2 = torch.cross(self.table.n, t1)
        self.t1 = t1
        self.t2 = t2  # (3,)

    def _sample_offset(self, bs: int) -> torch.Tensor:
        """
        Returns (bs, 3) offset vectors within the table footprint.
        We try up to 10 times; fall back to nearest in-footprint point.
        """
        dev = self.t1.device
        p0 = self._ref_transl  # (bs, 3)
        offsets = torch.zeros(bs, 3, device=dev)
        for b in range(bs):
            for _ in range(10):
                r = torch.randn(2, device=dev) * self.offset_std
                cand = p0[b] + r[0] * self.t1 + r[1] * self.t2
                if self.table._inside_footprint(cand[None]).item():
                    offsets[b] = cand - p0[b]
                    break
            else:
                # clip to nearest point inside
                d = self.table.distance_to_surface(p0[b])
                proj = p0[b] - d * self.table.n
                offsets[b] = proj - p0[b]
        return offsets  # (bs, 3)

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,
        poses: torch.Tensor,
        obj_verts: torch.Tensor,
    ) -> torch.Tensor:
        bs = trans.size(0)
        device = trans.device

        # first invocation → store reference pose & offset
        if self._ref_transl is None:
            self._ref_transl = trans[:, 0, :].detach()  # (bs, 3)
            self._rand_offset = self._sample_offset(bs)  # (bs, 3)

        frames = torch.arange(trans.shape[1], device=device)
        mask = frames >= cur_start_frame
        mask = mask & torch.isin(frames, self.target_frames)
        if not mask.any():
            return torch.zeros(bs, device=device)

        tgt_transl = self._ref_transl + self._rand_offset  # (bs, 3)
        trans_masked = trans[:, mask, :]  # (bs, F, 3)
        offset_loss = (trans_masked - tgt_transl[:, None]).pow(2).mean(dim=(1, 2))

        verts = obj_verts[:, mask]  # (bs, F, V, 3)
        signed = self.table.distance_to_surface(verts)  # (bs, F, V)
        k = min(self.k_lowest, signed.size(2))
        lowest_k = torch.topk(signed, k, dim=2, largest=False).values
        contact_loss = lowest_k.abs().mean(dim=(1, 2))
        penetration_loss = self.table.loss_below_surface(verts).mean(dim=(1, 2))
        outside_loss = self.table.loss_outside_polygon(verts).mean(dim=(1, 2))
        smooth_loss = zero_movement_loss(trans_masked)

        loss = (
            self.w_offset * offset_loss
            + self.w_contact * contact_loss
            + self.w_penetr * penetration_loss
            + self.w_outside * outside_loss
            + self.w_smooth * smooth_loss
        )
        return loss


def get_geometries(model_kwargs, b, n_points: int = 1000):
    # object
    object_name = model_kwargs["y"]["metadata"][b]["object_name"]
    obj_v_template, obj_faces = load_mesh(object_name, n_simplify_faces=None)
    v_inds = np.random.choice(obj_v_template.shape[0], n_points, replace=False)
    obj_v_template = obj_v_template[v_inds]
    obj_v_template = torch.tensor(obj_v_template, dtype=torch.float32)

    # table
    grab_seq_path = model_kwargs["y"]["metadata"][b]["grab_seq_path"]
    table_faces, table_verts, table_corner_locs = get_table_params(grab_seq_path)
    return {
        "obj_v_template": obj_v_template,
        "table_faces": table_faces,
        "table_verts": table_verts,
        "table_corner_locs": table_corner_locs,
    }


def get_geoms_batch(model_kwargs):
    results = defaultdict(list)
    for b in range(model_kwargs["y"]["prefix"].shape[0]):
        for k, v in get_geometries(model_kwargs, b).items():
            results[k].append(v)
    return results


def find_contiguous_static_blocks(contact_mask: torch.Tensor) -> List[Tuple[int, int]]:
    """Find contiguous blocks of frames where there is no contact.

    Args:
        contact_mask: (seq_len,) boolean tensor indicating contact

    Returns:
        List of (start_idx, end_idx) tuples for each contiguous static block
    """
    static_blocks = []
    seq_len = contact_mask.shape[0]
    start_idx = None

    for i in range(seq_len):
        if not contact_mask[i]:  # No contact
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:  # Contact found after no-contact
            static_blocks.append((start_idx, i))
            start_idx = None

    # Handle case where last block extends to end
    if start_idx is not None:
        static_blocks.append((start_idx, seq_len))

    return static_blocks


class KeepObjectStaticLoss(nn.Module):
    def __init__(
        self,
        th_contact_force=0.5,
    ):
        super().__init__()
        self.th_contact_force = th_contact_force

    def compute_static_loss(
        self, poses: torch.Tensor, trans: torch.Tensor, contact: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss to enforce object constancy during no-contact periods for one representation.

        Args:
            poses: (B, seq_len, 3) object pose rotations
            trans: (B, seq_len, 3) object translations
            contact: (B, seq, n_anchors) contact forces

        Returns:
            Tensor: (B,) loss values per batch
        """
        batch_size = poses.shape[0]

        # Compute frames with any contact for each hand separately
        has_contact = (contact > self.th_contact_force).any(dim=2)  # (B, seq)

        # static_frames = ~has_contact

        loss = torch.zeros(batch_size, device=poses.device)
        for b in range(batch_size):
            batch_loss = torch.tensor(0.0, device=poses.device)
            static_blocks = find_contiguous_static_blocks(has_contact[b])

            for start_idx, end_idx in static_blocks:
                if end_idx - start_idx < 2:  # Skip single frame blocks
                    continue

                # Get block of frames
                block_poses = poses[b, start_idx:end_idx]
                block_trans = trans[b, start_idx:end_idx]

                block_poses_dif = block_poses - block_poses[0].expand_as(
                    block_poses
                )  # (block_len-1, 3)
                block_trans_dif = block_trans - block_trans[0].expand_as(
                    block_trans
                )  # (block_len-1, 3)
                pose_diff = torch.abs(block_poses_dif).mean()
                trans_diff = torch.abs(block_trans_dif).mean()

                # Sum up differences across the block
                block_loss = pose_diff + trans_diff
                batch_loss += block_loss

            loss[b] = batch_loss

        return loss

    def __call__(
        self,
        cur_start_frame: int,
        trans: torch.Tensor,
        poses: torch.Tensor,
        obj_verts: torch.Tensor,
        smpldata_lst: Optional[List[SmplData]],
    ) -> torch.Tensor:
        """
        Args:
            contact_records: List[ContactRecord] of length B
            obj_verts: (B, n_verts, 3) object vertices
            smpldata_lst: List[SmplData] of length B containing object poses/translations
        """
        # Stack contact forces from both hands
        contact = torch.stack(
            [sd.contact for sd in smpldata_lst]
        )  # (B, seq, n_anchors)

        # Get object poses and translations for all representations
        poses_obj = torch.stack(
            [smpldata.poses_obj for smpldata in smpldata_lst]
        )  # (B, seq, 3)
        trans_obj = torch.stack(
            [smpldata.trans_obj for smpldata in smpldata_lst]
        )  # (B, seq, 3)

        # Compute static loss for each representation
        loss = self.compute_static_loss(poses_obj, trans_obj, contact)
        return loss
