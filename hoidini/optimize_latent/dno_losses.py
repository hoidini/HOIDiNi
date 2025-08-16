from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import torch
from torch import Tensor
from typing import Any, Callable, List, Optional

from hoidini.datasets.grab.grab_object_records import ContactRecord
from hoidini.cphoi.cphoi_utils import FeaturesDecoderWrapper, smpldata_to_contact_pairs
from hoidini.object_contact_prediction.cpdm_dno_conds import (
    AboveTableLoss,
    BatchedObjectModel,
    KeepObjectStaticLoss,
    Similar6DoFLoss,
)
from hoidini.object_contact_prediction.cpdm_dataset import ContactPairsSequence
from hoidini.objects_fk import ObjectModel
from hoidini.optimize_latent.dno_loss_functions import (
    FootSkateLoss,
    HandsContactLoss,
    HoiAboveTableLoss,
    HoiSideTableLoss,
    JointsJitterLoss,
    get_contact_record_from_smpldata,
)
from hoidini.datasets.smpldata import SmplModelsFK
from hoidini.geometry3d.hands_intersection_loss import HandIntersectionLoss


class DnoLossSource(Enum):
    FROM_NN = "from_nn"  # aka nearest neighbor
    FROM_CONTACT_PAIRS = "from_contact_pairs"
    FROM_CONTACT_PAIRS_ONE_PHASE = "from_contact_pairs_one_phase"


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
    "AboveTableLoss": {
        "_class_": AboveTableLoss,
        "cur_start_frame": "cur_start_frame",
        "poses": "poses_obj_full",
        "trans": "trans_obj_full",
        "obj_verts": "obj_verts_full",
        "smpldata_lst": "smpldata_lst_full",
    },
    "Similar6DoFLoss": {
        "_class_": Similar6DoFLoss,
        "cur_start_frame": "cur_start_frame",
        "trans": "trans_obj_full",
        "poses": "poses_obj_full",
        "obj_verts": "obj_verts_full",
        "smpldata_lst": "smpldata_lst_full",
    },
    "KeepObjectStaticLoss": {
        "_class_": KeepObjectStaticLoss,
        "cur_start_frame": "cur_start_frame",
        "trans": "trans_obj_full",
        "poses": "poses_obj_full",
        "obj_verts": "obj_verts_full",
        "smpldata_lst": "smpldata_lst_full",
    },
}


@dataclass
class DnoLossComponent:
    name: str
    weight: float
    loss_fn: Callable


class DnoCondition:
    """
    Will be used for both phase 1 and phase 2.
    """

    def __init__(
        self,
        decoder_wrapper: FeaturesDecoderWrapper,
        loss_components: List[DnoLossComponent],
        obj_model: BatchedObjectModel,
        smpl_fk: SmplModelsFK,
        pred_len: int,
        obj_verts_lst: Optional[List[Tensor]] = None,
        obj_faces_lst: Optional[List[Tensor]] = None,
        obj_fk_lst: Optional[List[ObjectModel]] = None,
        dno_loss_source: DnoLossSource = DnoLossSource.FROM_CONTACT_PAIRS,
        nearest_neighbor_per_frame: bool = True,  # related to the prediction of contact records
    ):
        self.decoder_wrapper = decoder_wrapper
        self.global_prefix_lst = None
        self.loss_components = loss_components
        self.dno_loss_source = dno_loss_source
        self.obj_model = obj_model
        self.smpl_fk = smpl_fk
        self.pred_len = pred_len
        self.obj_verts_lst = obj_verts_lst
        self.obj_faces_lst = obj_faces_lst
        self.obj_fk_lst = obj_fk_lst
        self.nearest_neighbor_per_frame = nearest_neighbor_per_frame

    def reset(self):
        self.global_prefix_lst = None

    def set_global_prefix_lst(self, global_prefix_lst: List[torch.Tensor]):
        self.global_prefix_lst = global_prefix_lst

    def set_current_seq(
        self, current_seq_lst: List[ContactRecord | ContactPairsSequence | Any]
    ):
        if isinstance(current_seq_lst[0], ContactRecord):
            self.contact_record_lst_current = current_seq_lst
        elif isinstance(current_seq_lst[0], ContactPairsSequence):
            self.contact_pairs_lst_current = current_seq_lst
        else:
            raise ValueError(f"Invalid sequence type: {type(current_seq_lst)}")

    def get_optimization_targets(self, smpldata_lst, body_verts):
        if self.dno_loss_source == DnoLossSource.FROM_NN:
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
        elif self.dno_loss_source in [
            DnoLossSource.FROM_CONTACT_PAIRS,
            DnoLossSource.FROM_CONTACT_PAIRS_ONE_PHASE,
        ]:
            if self.dno_loss_source == DnoLossSource.FROM_CONTACT_PAIRS:
                contact_pairs_lst: List[ContactPairsSequence] = (
                    self.contact_pairs_lst_current
                )
            else:
                contact_pairs_lst: List[ContactPairsSequence] = (
                    smpldata_to_contact_pairs(smpldata_lst)
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

    def __call__(self, samples: torch.Tensor):
        _, _, _, cur_len = samples.shape
        assert cur_len == self.pred_len
        assert samples.dim() == 4

        samples_full = torch.cat(self.global_prefix_lst + [samples], dim=3)
        full_len = samples_full.shape[3]
        cur_start_frame = full_len - cur_len

        # decode smpldata from features
        smpldata_lst_full = self.decoder_wrapper.decode(samples_full)
        smpldata_lst_cur = [
            smpldata.cut(cur_start_frame, None) for smpldata in smpldata_lst_full
        ]

        loss_fn_inputs = {
            "smpldata_lst_full": smpldata_lst_full,
            "cur_start_frame": cur_start_frame,
        }

        ############
        # Object related
        ############
        if self.obj_model is not None:
            poses_obj_full = torch.stack(
                [smpldata.poses_obj for smpldata in smpldata_lst_full]
            )  # (batch, seq_len, 3)
            trans_obj_full = torch.stack(
                [smpldata.trans_obj for smpldata in smpldata_lst_full]
            )  # (batch, seq_len, 3)
            obj_verts_full = self.obj_model(poses_obj_full, trans_obj_full)

            loss_fn_inputs.update(
                {
                    "obj_verts_full": obj_verts_full,
                    "poses_obj_full": poses_obj_full,
                    "trans_obj_full": trans_obj_full,
                }
            )

        ############
        # Human related
        ############
        if self.smpl_fk is not None:
            # Run FK on human only on the current sequence + one frame before (for continuity loss)
            smpldata_lst_cur_p1 = [
                smpl_data.cut(cur_start_frame - 1, None)
                for smpl_data in smpldata_lst_full
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
            opt_tgts = self.get_optimization_targets(smpldata_lst_cur, body_verts_cur)

            loss_fn_inputs.update(
                {
                    "contact_force_lhand": opt_tgts["contact_force_lhand"],
                    "contact_force_rhand": opt_tgts["contact_force_rhand"],
                    "tgt_lhand_verts": opt_tgts["tgt_lhand_verts"],
                    "tgt_rhand_verts": opt_tgts["tgt_rhand_verts"],
                    "object_verts_lst": opt_tgts["object_verts_lst"],
                    "object_faces_lst": opt_tgts["object_faces_lst"],
                    "src_body_verts": body_verts_cur,
                    "body_verts": body_verts_cur,
                    "joints_cur_p1": joints_cur_p1,
                }
            )

        loss = torch.zeros(len(samples), device=samples.device)
        loss_dict = {}
        for dno_loss_component in self.loss_components:
            loss_fn_cls_name = type(dno_loss_component.loss_fn).__name__
            mapping = LOSS_FUNCTIONS_ARGS_WIRING[loss_fn_cls_name]
            kwargs = {
                k: loss_fn_inputs[v]
                for k, v in mapping.items()
                if not k.startswith("_")
            }
            cur_loss = dno_loss_component.loss_fn(**kwargs)
            if torch.isnan(cur_loss).any():
                print(f"NaN loss for {dno_loss_component.name}")
                print(cur_loss)
                raise ValueError(f"NaN loss for {dno_loss_component.name}")

            loss += dno_loss_component.weight * cur_loss
            loss_dict[dno_loss_component.name] = cur_loss.clone().detach()
        return loss, loss_dict
