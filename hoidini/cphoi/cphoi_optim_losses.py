from collections import defaultdict
from hoidini.amasstools.geometry import axis_angle_to_matrix
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.cphoi.cphoi_utils import FeaturesDecoderWrapper
from hoidini.datasets.grab.grab_utils import load_mesh
from hoidini.datasets.smpldata import SmplData
from hoidini.object_contact_prediction.cpdm_dno_conds import BatchedObjectModel, DnoLoss
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from hoidini.datasets.grab.grab_utils import get_table_params, load_mesh


def get_cphoi_geometries_batch(model_kwargs, n_simplify_object: int = 3000):
    results = defaultdict(list)
    for b in range(model_kwargs["y"]["prefix"].shape[0]):
        for k, v in get_cphoi_geometries(
            model_kwargs, b, n_simplify_faces=n_simplify_object
        ).items():
            results[k].append(v)
    return results


def get_cphoi_geometries(model_kwargs, b, n_simplify_faces: int = 3000):
    # object
    object_name = model_kwargs["y"]["metadata"][b]["object_name"]
    obj_v_template, obj_faces = load_mesh(
        object_name, n_simplify_faces=n_simplify_faces
    )
    # obj_v_template = obj_v_template[v_inds]

    # table
    grab_seq_path = model_kwargs["y"]["metadata"][b]["grab_seq_path"]
    table_faces, table_verts, table_corner_locs = get_table_params(grab_seq_path)
    return {
        "obj_v_template": obj_v_template,
        "obj_faces": obj_faces,
        "table_faces": table_faces,
        "table_verts": table_verts,
        "table_corner_locs": table_corner_locs,
    }


class CphoiObjectDnoCond:
    def __init__(
        self,
        decoder_wrapper: FeaturesDecoderWrapper,
        dno_losses: Dict[str, Tuple[float, DnoLoss]],
        obj_v_template_batch: np.ndarray,
    ):
        self.decoder_wrapper = decoder_wrapper
        self.global_prefix_lst = None
        self.dno_losses = dno_losses
        self.obj_model = BatchedObjectModel(obj_v_template_batch)

    def __call__(self, samples: torch.Tensor):
        cur_len = samples.shape[3]
        if self.global_prefix_lst is not None:
            samples_full = torch.cat(self.global_prefix_lst + [samples], dim=3)
        else:
            samples_full = samples
        full_len = samples_full.shape[3]
        cur_start_frame = full_len - cur_len
        smpldata_lst = self.decoder_wrapper.decode(samples_full)
        poses_full = torch.stack(
            [sd.poses_obj for sd in smpldata_lst]
        )  # (batch, seq_len, 3)
        trans_full = torch.stack(
            [sd.trans_obj for sd in smpldata_lst]
        )  # (batch, seq_len, 3)
        obj_verts_full = self.obj_model(poses_full, trans_full)

        loss = torch.zeros(len(samples), device=samples.device)
        loss_dict = {}
        for k, (w, loss_fn) in self.dno_losses.items():
            cur_loss = loss_fn(
                cur_start_frame, trans_full, poses_full, obj_verts_full, smpldata_lst
            )
            if torch.isnan(cur_loss).any():
                print(f"NaN loss for {k}")
                print(cur_loss)
                raise ValueError(f"NaN loss for {k}")
            loss += w * cur_loss
            loss_dict[k] = cur_loss.clone().detach()
        return loss, loss_dict

    def reset(self):
        self.global_prefix_lst = None

    def set_global_prefix_lst(self, global_prefix_lst: List[torch.Tensor]):
        self.global_prefix_lst = global_prefix_lst
