from copy import deepcopy
from typing import List
import numpy as np
import torch

from hoidini.amasstools.geometry import axis_angle_to_matrix, matrix_to_axis_angle
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.datasets.dataset_smplrifke import samples_to_smpldata_lst
from hoidini.datasets.grab.grab_utils import get_MANO_SMPLX_vertex_ids
from hoidini.datasets.smpldata import SmplData, SmplModelsFK
from hoidini.normalizer import Normalizer
from hoidini.object_contact_prediction.cpdm_dataset import (
    ContactPairsSequence,
    FeatureProcessor,
)
from hoidini.objects_fk import ObjectModel
from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info


class FeaturesDecoderWrapper:
    def __init__(
        self,
        feature_processor: FeatureProcessor,
        normalizer: Normalizer,
        tfm_processor: torch.Tensor,
    ):
        self.feature_processor = feature_processor
        self.normalizer = normalizer
        self.tfm_processor = tfm_processor

    def decode(self, features) -> List[SmplData]:
        smpldata_lst = samples_to_smpldata_lst(
            features, self.normalizer, self.feature_processor, self.tfm_processor
        )
        return smpldata_lst


def smpldata_to_contact_pairs(
    smpldata_lst: List[SmplData] | SmplData,
) -> List[ContactPairsSequence] | ContactPairsSequence:
    is_single = isinstance(smpldata_lst, SmplData)
    if is_single:
        smpldata_lst = [smpldata_lst]
    contact_pairs_lst = []
    for smpldata in smpldata_lst:
        contact_pairs = ContactPairsSequence(
            local_object_points=smpldata.local_object_points,
            contacts=smpldata.contact,
            object_poses=smpldata.poses_obj,
            object_trans=smpldata.trans_obj,
        )
        contact_pairs_lst.append(contact_pairs)
    if is_single:
        return contact_pairs_lst[0]
    else:
        return contact_pairs_lst


def extract_local_obj_points_w_nn(
    smpldata_lst: List[SmplData], obj_v_template_lst: List[np.ndarray]
):
    bs = len(smpldata_lst)
    mano_smplx_vertex_ids = get_MANO_SMPLX_vertex_ids()
    anchor_inds_R2hands, _, _ = get_contact_anchors_info()
    smpldata_new_lst = []
    for b in range(bs):
        smpldata = smpldata_lst[b]
        obj_v_template = obj_v_template_lst[b]
        obj_v_template_torch = torch.from_numpy(obj_v_template).to(dist_util.dev())
        smpl_fk = SmplModelsFK.create("smplx", len(smpldata), device=dist_util.dev())
        obj_fk = ObjectModel(obj_v_template, len(smpldata)).to(dist_util.dev())
        with torch.no_grad():
            body_verts = smpl_fk.smpldata_to_smpl_output(smpldata).vertices
            obj_verts = obj_fk(
                transl=smpldata.poses_obj, global_orient=smpldata.trans_obj
            ).vertices.clone()
        contact_per_hand = smpldata.get_contact_per_hand()
        smpldata_new = deepcopy(smpldata)
        for hand in ["left", "right"]:
            hand_verts_seq = body_verts[:, mano_smplx_vertex_ids[hand + "_hand"], :]
            anchor_verts_seq = hand_verts_seq[:, anchor_inds_R2hands]
            contact = contact_per_hand[hand]
            contact = contact > 0.5
            # Compute distances and closest vertices without in-place operations
            dists = torch.cdist(
                anchor_verts_seq, obj_verts
            )  # (seq_len, n_anchors, n_obj_verts)
            closest_obj_vert_inds = torch.argmin(dists, dim=2)  # (seq_len, n_anchors)
            # local_object_points_seq = torch.gather(obj_verts, dim=1, index=closest_obj_vert_inds.unsqueeze(-1).expand(-1, -1, 3))  # (seq_len, n_anchors, 3)
            local_object_points = obj_v_template_torch[closest_obj_vert_inds]
            contact_inds = smpldata.get_contact_inds_per_hand()[hand]
            smpldata_new.local_object_points[:, contact_inds] = local_object_points

        smpldata_new_lst.append(smpldata_new)
    return smpldata_new_lst


def transform_smpldata(
    smpldata: SmplData, rot_mat: torch.Tensor, trans_xyz: torch.Tensor
):
    """
    Only transform the body and object.
    """
    assert rot_mat.shape == (3, 3)
    assert trans_xyz.shape == (3,)

    seq_len = len(smpldata)
    rot_mat = rot_mat.expand(seq_len, -1, -1)
    trans_xyz = trans_xyz.unsqueeze(0).expand(seq_len, -1)
    smpldata_new = deepcopy(smpldata)

    # body
    smpldata_new.poses[:, :3] = matrix_to_axis_angle(
        rot_mat @ axis_angle_to_matrix(smpldata_new.poses[:, :3])
    )
    smpldata_new.trans = (
        torch.einsum("bij,bj->bi", rot_mat, smpldata_new.trans) + trans_xyz
    )
    smpldata_new.joints = torch.einsum(
        "bij,bkj->bki", rot_mat, smpldata_new.joints
    ) + trans_xyz.unsqueeze(1).expand(
        len(smpldata_new), smpldata_new.joints.shape[1], 3
    )

    # object (if present)
    smpldata_new.poses_obj = matrix_to_axis_angle(
        rot_mat.transpose(1, 2) @ axis_angle_to_matrix(smpldata_new.poses_obj)
    )
    smpldata_new.trans_obj = (
        torch.einsum("bij,bj->bi", rot_mat, smpldata_new.trans_obj) + trans_xyz
    )
    return smpldata_new
