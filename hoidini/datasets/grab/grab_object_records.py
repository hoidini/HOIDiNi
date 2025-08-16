from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict
import numpy as np
import trimesh
import pickle
import os
from torch import Tensor
import torch

import smplx
from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info
from hoidini.datasets.grab.grab_utils import (
    get_MANO_SMPLX_vertex_ids,
    grab_seq_data_to_object_name,
)
from hoidini.datasets.grab.grab_utils import (
    parse_npz,
    simplify_trimesh,
    reduce_seq_data,
    params2torch,
    GRAB_DATA_PATH,
)
from hoidini.objects_fk import ObjectModel
from hoidini.datasets.smpldata import SMPL_MODELS_PATH
from hoidini.general_utils import torchify_numpy_dict


def extract_grab_contact_data(
    data_path: str,
    tgt_fps: int = 20,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    n_simplify_object: int = 4000,
) -> Dict[str, Tensor]:
    """Create a GrabContactRecord from a GRAB dataset sequence file.
    Args:
        data_path: Path to the .npz sequence file
        tgt_fps: Target frames per second
        start_frame: Start frame
        end_frame: End frame
    Returns:
        GrabContactRecord containing the object template and contact information
    """
    seq_data = parse_npz(data_path)
    seq_data = reduce_seq_data(seq_data, tgt_fps, start_frame, end_frame)

    n_comps = seq_data["n_comps"]
    gender = seq_data["gender"]
    T = seq_data["n_frames"]

    lhand_mesh_path = os.path.join(
        GRAB_DATA_PATH, seq_data["gender"], os.path.basename(seq_data["lhand"]["vtemp"])
    )
    rhand_mesh_path = os.path.join(
        GRAB_DATA_PATH, seq_data["gender"], os.path.basename(seq_data["rhand"]["vtemp"])
    )
    lhand_betas_path = lhand_mesh_path.replace(".ply", "_betas.npy")
    rhand_betas_path = rhand_mesh_path.replace(".ply", "_betas.npy")

    hand_betas = {"left": np.load(lhand_betas_path), "right": np.load(rhand_betas_path)}
    hand_keys = {"left": "lhand", "right": "rhand"}
    hand_mesh_paths = {"left": lhand_mesh_path, "right": rhand_mesh_path}

    use_betas = False
    use_betas_zero = False

    obj_mesh_path = os.path.join(
        GRAB_DATA_PATH,
        os.path.relpath(
            seq_data["object"]["object_mesh"],
            os.path.dirname(os.path.dirname(seq_data["object"]["object_mesh"])),
        ),
    )
    # table_mesh_path = os.path.join(grab_data_path, os.path.relpath(seq_data['table']['table_mesh'], os.path.dirname(os.path.dirname(seq_data['table']['table_mesh']))))

    object_name = grab_seq_data_to_object_name(seq_data)

    hands_verts = {}
    hands_faces = {}

    for hand in ["left", "right"]:
        if use_betas:
            hand_mano_model_path = os.path.join(
                SMPL_MODELS_PATH, f"mano/MANO_{hand.upper()}.pkl"
            )
            with open(hand_mano_model_path, "rb") as f:
                d = pickle.load(f, encoding="latin")
            mesh_faces = d["f"]
            mesh_vertices = d["v_template"]
        else:
            mesh = trimesh.load(hand_mesh_paths[hand])
            mesh_faces = np.array(mesh.faces)
            mesh_vertices = np.array(mesh.vertices)

        smplx.MANO.SHAPE_SPACE_DIM = 10  # to disable warning message
        hand_model: smplx.MANO = smplx.create(
            model_path=SMPL_MODELS_PATH,
            is_rhand=hand == "right",
            model_type="mano",
            gender=gender,
            flat_hand_mean=True,
            num_pca_comps=n_comps,
            v_template=mesh_vertices if not use_betas else None,
            batch_size=T,
        )
        hand_model.SHAPE_SPACE_DIM = 10  # to disable warning message
        hand_key = hand_keys[hand]
        hand_params = params2torch(seq_data[hand_key]["params"])

        if use_betas:
            betas = hand_betas[hand]
            betas = np.tile(betas, (T, 1))
            betas = torch.from_numpy(betas).float()
            if use_betas_zero:
                betas = torch.zeros_like(betas)
        else:
            betas = None

        mano_output: smplx.MANO.forward = hand_model(
            betas=betas,
            global_orient=hand_params["global_orient"],
            hand_pose=hand_params["hand_pose"],
            transl=hand_params["transl"],
        )
        hand_verts_anim = mano_output.vertices.detach().cpu().numpy()

        hands_verts[hand] = hand_verts_anim
        hands_faces[hand] = mesh_faces
        # animate_mesh(f"Mano_{hand}", mesh_faces, hand_verts_anim)

    # object
    obj_mesh = trimesh.load(obj_mesh_path)
    print(f"simplifying object mesh to {n_simplify_object} faces")
    obj_mesh = simplify_trimesh(obj_mesh, tgt_faces=n_simplify_object)
    obj_faces = np.array(obj_mesh.faces)
    obj_v_template = np.array(obj_mesh.vertices)
    obj_model = ObjectModel(v_template=obj_v_template, batch_size=T)
    obj_params = params2torch(seq_data["object"]["params"])
    obj_verts = obj_model(**obj_params).vertices.detach().cpu().numpy()
    # animate_mesh("Object", obj_faces, obj_verts_anim)

    mano_smplx_vertex_ids = get_MANO_SMPLX_vertex_ids()

    # Extract `contact` locations for each hand
    contact_body = seq_data["contact"]["body"]
    lhand_contact_force = contact_body[:, mano_smplx_vertex_ids["left_hand"]]
    rhand_contact_force = contact_body[:, mano_smplx_vertex_ids["right_hand"]]

    return {
        "obj_verts": torch.from_numpy(obj_verts),
        "obj_faces": torch.from_numpy(obj_faces),
        "lhand_vert_locs": torch.from_numpy(hands_verts["left"]),
        "rhand_vert_locs": torch.from_numpy(hands_verts["right"]),
        "lhand_contact_force": torch.from_numpy(lhand_contact_force),
        "rhand_contact_force": torch.from_numpy(rhand_contact_force),
        "obj_trans": obj_params["transl"],
        "obj_poses": obj_params["global_orient"],
        "obj_v_template": torch.from_numpy(obj_v_template).to(torch.float32),
        "object_name": object_name,
    }


@dataclass
class ContactRecord:
    # for penetration loss
    obj_verts: Tensor  # (seq_len, n_verts, 3)
    obj_faces: Tensor  # (n_faces, 3)

    # for contact loss
    lhand_vert_locs: Tensor  # (seq_len, n_verts, 3)
    rhand_vert_locs: Tensor  # (seq_len, n_verts, 3)
    lhand_contact_force: Tensor  # (seq_len, n_verts)
    rhand_contact_force: Tensor  # (seq_len, n_verts)

    def __len__(self):
        return self.obj_verts.shape[0]

    def reduce_contact_to_anchors(self):
        """
        Reduce contact vertices to anchor vertices.
        Anchor vertex will receive the sum of it's non anchor neighbors and the non anchors neighbors will be zeroed
        """
        anchor_inds_R2hands, closest_anchor_per_vertex_R2anchors, _ = (
            get_contact_anchors_info()
        )
        for hand in ["left", "right"]:
            hand_contact_force_old = (
                self.lhand_contact_force if hand == "left" else self.rhand_contact_force
            )
            hand_contact_force_new = torch.zeros_like(hand_contact_force_old).to(
                dtype=torch.int16
            )
            for anchor_ind_R2anchors, anchor_ind_R2hands in enumerate(
                anchor_inds_R2hands
            ):
                mask = closest_anchor_per_vertex_R2anchors == anchor_ind_R2anchors
                hand_contact_force_new[:, anchor_ind_R2hands] = hand_contact_force_old[
                    :, mask
                ].sum(dim=1)

            assert hand_contact_force_new.shape == hand_contact_force_old.shape
            if hand == "left":
                self.lhand_contact_force = hand_contact_force_new
            else:
                self.rhand_contact_force = hand_contact_force_new

    @classmethod
    def get_sequential_attr_names(cls):
        """
        Returns a list of the sequential elements of the contact record.
        """
        return [
            "obj_verts",
            "lhand_vert_locs",
            "rhand_vert_locs",
            "lhand_contact_force",
            "rhand_contact_force",
        ]

    @classmethod
    def get_spatial_attr_names(cls):
        return [
            "obj_verts",
            "lhand_vert_locs",
            "rhand_vert_locs",
        ]

    def to(self, device: torch.device) -> ContactRecord:
        new_values = {field: value.to(device) for field, value in asdict(self).items()}
        return ContactRecord(**new_values)

    def cut(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ) -> ContactRecord:
        d = asdict(self)
        seq_attr_names = self.get_sequential_attr_names()
        d = {
            k: v[start_frame:end_frame] if k in seq_attr_names else v
            for k, v in d.items()
        }
        return ContactRecord(**d)

    def translate(self, translation: Tensor) -> ContactRecord:
        d = asdict(self)
        translation = translation.flatten()
        spatial_attr_names = self.get_spatial_attr_names()
        d = {k: v + translation if k in spatial_attr_names else v for k, v in d.items()}
        return ContactRecord(**d)

    def save(self, path: str) -> None:
        data_dict = {k: v.numpy() for k, v in asdict(self).items()}
        np.savez(path, **data_dict)

    @staticmethod
    def load(path: str) -> ContactRecord:
        """Load a GrabContactRecord from a .npz file at the given path"""
        data = np.load(path)
        data = torchify_numpy_dict(data)
        return ContactRecord(**data)

    # @classmethod
    # def from_obj_name(cls, obj_name: str, n_faces_obj: int = 400) -> ContactRecord:
    #     """
    #     Return empty contact record with object faces.
    #     """
    #     obj_mesh_path = os.path.join(GRAB_DATA_PATH, 'contact_meshes', obj_name)
    #     obj_mesh = trimesh.load(obj_mesh_path)
    #     print(f"simplifying object mesh to {n_faces_obj} faces")
    #     obj_mesh = simplify_trimesh(obj_mesh, tgt_faces=n_faces_obj)
    #     obj_faces = np.array(obj_mesh.faces)
    #     return ContactRecord(
    #         obj_verts=None,
    #         obj_faces=torch.from_numpy(obj_faces),
    #         lhand_vert_locs=None,
    #         rhand_vert_locs=None,
    #         lhand_contact_force=None,
    #         rhand_contact_force=None,
    #         )

    @classmethod
    def from_grab_data(
        cls,
        data_path: str,
        tgt_fps: int = 20,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        n_simplify_object: int = 400,
    ):
        """Create a GrabContactRecord from a GRAB dataset sequence file.
        Args:
            data_path: Path to the .npz sequence file
            tgt_fps: Target frames per second
            start_frame: Start frame
            end_frame: End frame
        Returns:
            GrabContactRecord containing the object template and contact information
        """
        grab_contact_data = extract_grab_contact_data(
            data_path, tgt_fps, start_frame, end_frame, n_simplify_object
        )
        # Create contact record
        return ContactRecord(
            obj_verts=grab_contact_data["obj_verts"],
            obj_faces=grab_contact_data["obj_faces"],
            lhand_vert_locs=grab_contact_data["lhand_vert_locs"],
            rhand_vert_locs=grab_contact_data["rhand_vert_locs"],
            lhand_contact_force=grab_contact_data["lhand_contact_force"],
            rhand_contact_force=grab_contact_data["rhand_contact_force"],
        )


if __name__ == "__main__":
    data_point_path = (
        "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB/s2/teapot_lift.npz"
    )
    grab_contact_record = ContactRecord.from_grab_data(data_point_path)

    print(grab_contact_record)
