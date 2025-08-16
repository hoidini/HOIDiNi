from __future__ import annotations
import inspect

from hoidini.smplx.utils import SMPLXOutput

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
from hoidini.amasstools.geometry import matrix_to_axis_angle
import os
from hoidini import smplx
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass, fields

from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.general_utils import torchify_numpy_dict
from hoidini.skeletons.smplx_52 import SMPLX_52_144_INDS, SMPLX_JOINT_NAMES_52
from hoidini.resource_paths import SMPL_MODELS_DATA


SMPL_MODELS_PATH = SMPL_MODELS_DATA


class SmplModelsFK(ABC):
    model_name = None
    n_joints = None

    @classmethod
    def create(cls, model_name, batch_size, device):
        if model_name == "smpl":
            return SmplFK(batch_size, device)
        elif model_name == "smplx":
            return SmplxFK(batch_size, device)
        else:
            raise ValueError()

    @property
    def device(self):
        return next(self.sbj_model.parameters()).device

    @abstractmethod
    def smpldata_to_smpl_output(self, smpl_data: SmplData):
        pass

    def smpldata_to_smpl_output_batch(
        self, smpldata_lst: List[SmplData]
    ) -> List[smplx.utils.SMPLOutput]:
        return [self.smpldata_to_smpl_output(e) for e in smpldata_lst]


def get_smpl_model_path(model_name, gender="neutral", smpl_models_path=None):
    assert model_name in ["smpl", "smplx", "mano"]
    assert gender in ["male", "female", "neutral"]
    smpl_models_path = (
        smpl_models_path or os.getenv("SMPLX_MODELS_PATH") or SMPL_MODELS_DATA
    )
    smpl_model_path = os.path.join(
        smpl_models_path,
        f"{model_name.lower()}/{model_name.upper()}_{gender.upper()}.pkl",
    )
    return smpl_model_path


class SmplFK(SmplModelsFK):
    model_name = "smpl"
    n_joints = 24

    def __init__(self, batch_size, device="cuda"):
        self.sbj_model = smplx.create(
            model_path=get_smpl_model_path("smpl"),
            model_type="smpl",
            gender="neutral",
            batch_size=batch_size,
        ).to(device, dtype=torch.float)

    def smpldata_to_smpl_output(
        self, smpl_data: SmplData, return_verts=True
    ) -> SMPLXOutput:
        body_params = {
            "global_orient": smpl_data.poses[
                :, :3
            ],  # global orientation is the rotation of the
            "body_pose": smpl_data.poses[
                :, 3 : smpl_data.n_joints * 3
            ],  # body pose not including the root, use, there are 24 joints, pelvis already used
            "transl": smpl_data.trans,
        }
        smpl_output: smplx.SMPL.forward = self.sbj_model(**body_params)
        smpl_output.joints = smpl_output.joints[:, :24, :]  # <--- Reduce to 24 joints
        return smpl_output


class SmplxFK(SmplModelsFK):
    model_name = "smplx"
    n_joints = 52

    def __init__(self, batch_size, device="cuda"):
        self.sbj_model: smplx.SMPLX = smplx.create(
            model_path=SMPL_MODELS_DATA,
            model_type="smplx",
            gender="neutral",
            create_jaw_pose=True,
            batch_size=batch_size,
            use_pca=False,
            flat_hand_mean=True,  # Double check this param
        ).to(device)

    def smpldata_to_smpl_output(
        self, smpldata: SmplData, return_verts=True, cancel_offset=True
    ) -> smplx.utils.SMPLXOutput:
        """
        cancel_offset: for some reason
        """
        poses = smpldata.poses
        body_params = {
            "transl": smpldata.trans,
            "global_orient": poses[:, :3],
            "body_pose": poses[:, 3 : 22 * 3],
            "left_hand_pose": poses[:, 22 * 3 : (22 + 15) * 3],
            "right_hand_pose": poses[:, (22 + 15) * 3 : (22 + 15 + 15) * 3],
            "return_verts": return_verts,
        }
        smplx_output: smplx.SMPLXOutput = self.sbj_model(**body_params)
        smplx_output.joints = smplx_output.joints[
            :, SMPLX_52_144_INDS, :
        ]  # <--- Reduce to 52 joints

        if cancel_offset:
            offset = smpldata.trans[0] - smplx_output.joints[0, 0]
            smplx_output.joints += offset
            smplx_output.vertices += offset

        return smplx_output


@dataclass
class SmplData:
    """
    Contains data that is used for the creation of smplrifke features
    Also being created from smplrifke features, in this case, for some attributes, fk is required
    Redundant data is included (e.g. joints) that requires extraction with forward kinematics
    """

    poses: Tensor  # (seq_len, n_joints, 3)
    trans: Tensor  # (seq_len, 3)
    joints: Tensor  # (seq_len, n_joints, 3)  # (for encode)

    joints_from_poses: Tensor = None  # (seq_len, n_joints, 3)  (from decode+fk)

    global_lhand_rotmat: Tensor = None  # (seq_len, 3, 3)  (for encode)
    global_rhand_rotmat: Tensor = None  # (seq_len, 3, 3)  (for encode)

    poses_obj: Tensor = None  # (seq_len, 3)
    trans_obj: Tensor = None  # (seq_len, 3)

    contact: Tensor = None  # (seq_len, n_contact_vertices)

    poses_obj_from_lhand: Tensor = None  # (seq_len, 3)  (from decode+fk)
    poses_obj_from_rhand: Tensor = None  # (seq_len, 3)  (from decode+fk)

    trans_obj_from_lhand: Tensor = None  # (seq_len, 3)  (from decode+fk)
    trans_obj_from_rhand: Tensor = None  # (seq_len, 3)  (from decode+fk)

    # raw outputs to be used with FK
    _trans_obj_rel2_lhand: Tensor = (
        None  # (seq_len, 3)     (from decode, to be used with fk)
    )
    _trans_obj_rel2_rhand: Tensor = (
        None  # (seq_len, 3)     (from decode, to be used with fk)
    )
    _rotmat_obj_rel2_lhand: Tensor = (
        None  # (seq_len, 3, 3)  (from decode, to be used with fk)
    )
    _rotmat_obj_rel2_rhand: Tensor = (
        None  # (seq_len, 3, 3)  (from decode, to be used with fk)
    )

    local_object_points: Tensor = None  # (seq_len, n_anchors, 3)

    def get_contact_inds_per_hand(self) -> Dict[str, Tensor]:
        n_anchors_per_hand = self.contact.shape[1] // 2
        inds = torch.arange(0, self.contact.shape[1])
        return {"left": inds[:n_anchors_per_hand], "right": inds[n_anchors_per_hand:]}

    def get_contact_per_hand(self) -> Dict[str, Tensor]:
        n_anchors_per_hand = self.contact.shape[1] // 2
        return {
            "left": self.contact[:, :n_anchors_per_hand],
            "right": self.contact[:, n_anchors_per_hand:],
        }

    def fill_in_using_fk(self, smplx_output: smplx.SMPLXOutput = None, fk_device=None):
        """
        Use forward kinematics to fill in the missing data the model doesn't predict
        """
        if smplx_output is None:
            fk_device = fk_device if fk_device is not None else dist_util.dev()
            smplx_fk = SmplxFK(batch_size=len(self), device=fk_device)
            smplx_output = smplx_fk.smpldata_to_smpl_output(self)

        self.joints_from_poses = smplx_output.joints
        self.global_lhand_rotmat = smplx_output.global_joints_transforms[
            :, SMPLX_JOINT_NAMES_52.index("left_wrist"), :3, :3
        ]
        self.global_rhand_rotmat = smplx_output.global_joints_transforms[
            :, SMPLX_JOINT_NAMES_52.index("right_wrist"), :3, :3
        ]

        # Left hand
        obj_rotmat_r2h_left = self._rotmat_obj_rel2_lhand
        obj_trans_r2h_left = self._trans_obj_rel2_lhand
        global_hand_rotmat_left = self.global_lhand_rotmat
        global_hand_trans_left = self.joints_from_poses[
            :, SMPLX_JOINT_NAMES_52.index("left_wrist")
        ]
        global_obj_rotmat_left = global_hand_rotmat_left @ obj_rotmat_r2h_left
        global_obj_trans_left = (
            torch.einsum("bij,bj->bi", global_hand_rotmat_left, obj_trans_r2h_left)
            + global_hand_trans_left
        )
        poses_obj_from_lhand = matrix_to_axis_angle(global_obj_rotmat_left)
        self.poses_obj_from_lhand = poses_obj_from_lhand
        self.trans_obj_from_lhand = global_obj_trans_left

        # Right hand
        obj_rotmat_r2h_right = self._rotmat_obj_rel2_rhand
        obj_trans_r2h_right = self._trans_obj_rel2_rhand
        global_hand_rotmat_right = self.global_rhand_rotmat
        global_hand_trans_right = self.joints_from_poses[
            :, SMPLX_JOINT_NAMES_52.index("right_wrist")
        ]
        global_obj_rotmat_right = global_hand_rotmat_right @ obj_rotmat_r2h_right
        global_obj_trans_right = (
            torch.einsum("bij,bj->bi", global_hand_rotmat_right, obj_trans_r2h_right)
            + global_hand_trans_right
        )
        poses_obj_from_rhand = matrix_to_axis_angle(global_obj_rotmat_right)
        self.poses_obj_from_rhand = poses_obj_from_rhand
        self.trans_obj_from_rhand = global_obj_trans_right

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def detach(self) -> SmplData:
        smpl_data_dict = self.to_dict()
        for key, value in smpl_data_dict.items():
            if value is not None:
                smpl_data_dict[key] = value.clone().detach()
        return SmplData(**smpl_data_dict)

    def to(self, *args, **kwargs) -> SmplData:
        smpl_data_dict = self.to_dict()
        for key, value in smpl_data_dict.items():
            if value is not None:
                smpl_data_dict[key] = value.to(*args, **kwargs)
        return SmplData(**smpl_data_dict)

    def __len__(self):
        return self.poses.shape[0]

    @property
    def n_joints(self):
        return self.joints.shape[1]

    def cut(self, start, end) -> SmplData:
        smpl_data_dict = self.to_dict()
        for key, value in smpl_data_dict.items():
            if value is not None:
                smpl_data_dict[key] = value[start:end]
        return SmplData(**smpl_data_dict)

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float):
        smpl_data_dict = np.load(path, allow_pickle=True)
        smpl_data_dict = torchify_numpy_dict(smpl_data_dict, device, dtype)
        return SmplData(**smpl_data_dict)


def slice_smpldata(smpl_data: SmplData, start: int, end: int) -> SmplData:
    """
    Returns a new SmplData object with all tensor attributes sliced from start to end along dim=0.
    Non-tensor attributes are left unchanged.
    """
    smpl_data_dict = smpl_data.to_dict()
    for key, value in smpl_data_dict.items():
        if isinstance(value, torch.Tensor):
            if end is None:
                smpl_data_dict[key] = value[start:]
            elif start is None:
                smpl_data_dict[key] = value[:end]

            else:
                smpl_data_dict[key] = value[start:end]
    return SmplData(**smpl_data_dict)


def upsample_smpldata(smpl_data: SmplData, factor) -> SmplData:
    """
    Upsample the smpl data by a factor of 2
    """
    smpl_data_dict = smpl_data.to_dict()
    T = len(smpl_data)
    up = torch.nn.Upsample(scale_factor=factor, mode="linear")
    for key, value in smpl_data_dict.items():
        if isinstance(value, torch.Tensor):
            smpl_data_dict[key] = (
                up(value.view(1, T, -1).permute(0, 2, 1))
                .permute(0, 2, 1)
                .view([-1] + list(value.shape[1:]))
            )
            assert smpl_data_dict[key].shape[0] == T * 2
            assert smpl_data_dict[key].shape[1:] == value.shape[1:]

    return SmplData(**smpl_data_dict)
