import json
import os
import pandas as pd
import torch
import numpy as np
import pickle
import trimesh
import open3d as o3d
from copy import deepcopy
from glob import glob
from typing import Dict, List, Optional
from torch import Tensor
from functools import lru_cache

from hoidini.general_utils import SRC_DIR
from hoidini.objects_fk import ObjectModel
from hoidini.resource_paths import GRAB_DATA_PATH, MANO_SMPLX_VERTEX_IDS_PATH


"""
##############################################
Grab data structure:
##############################################

body/vtemp  (path)

contact/body: (2310, 10475)
contact/object: (2310, 48370)

body/params/transl: (2310, 3)
body/params/global_orient: (2310, 3)
body/params/body_pose: (2310, 63)
body/params/jaw_pose: (2310, 3)
body/params/leye_pose: (2310, 3)
body/params/reye_pose: (2310, 3)
body/params/left_hand_pose: (2310, 24)
body/params/right_hand_pose: (2310, 24)
body/params/fullpose: (2310, 165)
body/params/expression: (2310, 10)

lhand/params/global_orient: (2310, 3)
lhand/params/hand_pose: (2310, 24)
lhand/params/transl: (2310, 3)
lhand/params/fullpose: (2310, 45)

rhand/params/global_orient: (2310, 3)
rhand/params/hand_pose: (2310, 24)
rhand/params/transl: (2310, 3)
rhand/params/fullpose: (2310, 45)

object/params/transl: (2310, 3)
object/params/global_orient: (2310, 3)

table/params/transl: (2310, 3)
table/params/global_orient: (2310, 3)
"""


def get_all_grab_seq_paths(grab_dataset_path: str = GRAB_DATA_PATH):
    all_seq_paths = sorted(glob("s*/*.npz", root_dir=grab_dataset_path))
    all_seq_paths = [os.path.join(grab_dataset_path, rp) for rp in all_seq_paths]
    return all_seq_paths


def _reduce_seq_data(data, skip_frames, start_frame, end_frame, parent_key=None):
    if isinstance(data, dict):
        return {
            k: _reduce_seq_data(v, skip_frames, start_frame, end_frame, parent_key=k)
            for k, v in data.items()
        }
    elif isinstance(data, np.ndarray) and parent_key is not None:
        if data.ndim >= 2 and data.shape[0] > skip_frames:
            return data[::skip_frames][start_frame:end_frame]
    return data


def reduce_seq_data(seq_data, tgt_fps, start_frame=None, end_frame=None):
    seq_data = deepcopy(seq_data)
    mocap_fps = seq_data["framerate"]
    if mocap_fps % tgt_fps != 0:
        raise ValueError(
            "The mocap framerate must be an integer multiple of the target framerate."
        )
    skip_frames = int(mocap_fps / tgt_fps)
    seq_data_reduced = _reduce_seq_data(seq_data, skip_frames, start_frame, end_frame)
    seq_data_reduced["n_frames"] = seq_data_reduced["body"]["params"][
        "body_pose"
    ].shape[0]
    seq_data_reduced["framerate"] = tgt_fps
    return seq_data_reduced


def parse_npz(npz_path, allow_pickle=True) -> Dict:
    d = dict(np.load(npz_path, allow_pickle=allow_pickle))
    return grab_data_to_proper_dict(d)


def translate_seq_data(seq_data, translation: np.ndarray | Tensor):
    """
    body/params/transl: (2310, 3)
    lhand/params/transl: (2310, 3)
    rhand/params/transl: (2310, 3)
    object/params/transl: (2310, 3)
    table/params/transl: (2310, 3)
    """
    if isinstance(translation, Tensor):
        translation = translation.detach().cpu().numpy()

    assert translation.shape == (3,)
    translation = translation.reshape(1, 3)
    seq_data = deepcopy(seq_data)
    seq_data["body"]["params"]["transl"] += translation
    seq_data["lhand"]["params"]["transl"] += translation
    seq_data["rhand"]["params"]["transl"] += translation
    seq_data["object"]["params"]["transl"] += translation
    seq_data["table"]["params"]["transl"] += translation
    return seq_data


def grab_data_to_proper_dict(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: grab_data_to_proper_dict(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [grab_data_to_proper_dict(item) for item in obj]
    else:
        return obj


def to_cpu(t):
    return t.detach().cpu().numpy()


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def simplify_trimesh(mesh, tgt_faces=500):
    o3d_mesh = o3d.geometry.TriangleMesh()
    # if isinstance(mesh, trimesh.scene.Scene):
    #     mesh = mesh.to_mesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    simplified_mesh = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=tgt_faces
    )
    simplified_trimesh = trimesh.Trimesh(
        vertices=np.asarray(simplified_mesh.vertices),
        faces=np.asarray(simplified_mesh.triangles),
    )
    return simplified_trimesh


@lru_cache()
def get_MANO_SMPLX_vertex_ids() -> Dict[str, np.ndarray]:
    """
    Returns array per hand of size ~700 (no. of mano mesh vertices)
    index loc: the mano vertex index
    index val: the smplx vertex index
    mano_smplx_vertex_ids["left_hand"] = np.array([v_l_id1, v_l_id2, ...])
    mano_smplx_vertex_ids["right_hand"] = np.array([v_r_id1, v_r_id2, ...])
    """
    with open(MANO_SMPLX_VERTEX_IDS_PATH, "rb") as f:
        mano_smplx_vertex_ids = pickle.load(f)
    return mano_smplx_vertex_ids


def grab_seq_path_to_seq_id(path):
    """
    path/to/grab/dataset/s4/camera_takepicture_3.npz ---> s4/camera_takepicture_3
    """
    parent_dir = os.path.basename(os.path.dirname(path))
    file_name = os.path.basename(path)
    seq_id = os.path.join(parent_dir, file_name).replace(".npz", "")
    return seq_id


def grab_seq_path_to_unique_name(grab_seq_path):
    """
    Useful for saving files with sequence ids
    """
    return grab_seq_path_to_seq_id(grab_seq_path).replace("/", "_")


def grab_object_name_to_path(object_name):
    object_name = object_name.replace(".ply", "")
    return os.path.join(GRAB_DATA_PATH, "contact_meshes", f"{object_name}.ply")


def grab_seq_data_to_object_name(seq_data: dict) -> str:
    return os.path.basename(seq_data["object"]["object_mesh"]).replace(".ply", "")


def grab_seq_path_to_object_name(grab_seq_path: str) -> str:
    return grab_seq_data_to_object_name(parse_npz(grab_seq_path))


def grab_seq_path_to_object_path(grab_seq_path: str) -> str:
    return grab_object_name_to_path(
        grab_seq_data_to_object_name(parse_npz(grab_seq_path))
    )


def grab_seq_data_to_object_path(seq_data: dict) -> str:
    return grab_object_name_to_path(grab_seq_data_to_object_name(seq_data))


def load_mesh(
    object_name: str, n_simplify_faces: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(object_name):
        path = object_name
    else:
        path = grab_object_name_to_path(object_name)
    obj_mesh = trimesh.load(path)
    if isinstance(obj_mesh, trimesh.scene.Scene):
        obj_mesh = obj_mesh.to_mesh()
    if n_simplify_faces is not None:
        obj_mesh = simplify_trimesh(obj_mesh, tgt_faces=n_simplify_faces)
    obj_verts = np.array(obj_mesh.vertices).astype(np.float32)
    obj_faces = np.array(obj_mesh.faces)
    return obj_verts, obj_faces


@lru_cache()
@lru_cache()
def get_df_prompts():
    df_prompts = pd.read_csv(
        os.path.join(SRC_DIR, "datasets/resources/grab_prompts.csv")
    ).set_index("dp_ind")
    df_prompts.index = df_prompts.index.str.replace(".npz", "")
    return df_prompts


@lru_cache()
@lru_cache()
def get_df_grab_index(tgt_fps: Optional[int] = None) -> pd.DataFrame:
    """
    Includes the following columns:
    seq_name, sbj_id, fps, n_frames(@fps), object_name, motion_intent
    """
    df_index = pd.read_csv(os.path.join(SRC_DIR, "datasets/resources/grab_index.csv"))
    if tgt_fps is not None:
        assert all(
            int(orig_fps) % tgt_fps == 0 for orig_fps in df_index["fps"]
        ), "All original fps must be divisible by the target fps"
        df_index["n_frames"] = df_index.apply(
            lambda row: int(row["n_frames"] * (tgt_fps / row["fps"])), axis=1
        )
        df_index["fps"] = tgt_fps
    df_index = df_index.set_index("seq_name")
    return df_index


def get_grab_split_ids(split: str) -> List[str]:
    base_dir = os.path.join(SRC_DIR, f"datasets/resources/grab_{split}_split.json")
    with open(base_dir, "r") as f:
        split_ids = json.load(f)
    return split_ids


def get_grab_split_seq_paths(
    split: str, grab_dataset_path: str = GRAB_DATA_PATH
) -> List[str]:
    split_ids = set(get_grab_split_ids(split))
    all_seq_paths = get_all_grab_seq_paths(grab_dataset_path)
    return [p for p in all_seq_paths if grab_seq_path_to_seq_id(p) in split_ids]


def grab_seq_id_to_seq_path(seq_id: str) -> str:
    return os.path.join(GRAB_DATA_PATH, seq_id + ".npz")


def get_table_params(grab_seq_path):
    TABLE_CORNER_VERT_IDS_SIDE1 = torch.tensor([15958, 3180, 2, 12765])
    TABLE_CORNER_VERT_IDS_SIDE2 = torch.tensor([5323, 8513, 21277, 18092])

    seq_data = parse_npz(grab_seq_path)  # no need to reduce fps, since it's static
    table_mesh_path = grab_object_name_to_path("table")
    table_mesh = trimesh.load(table_mesh_path)
    table_v_template = np.array(table_mesh.vertices)
    table_faces = np.array(table_mesh.faces)
    table_model = ObjectModel(v_template=table_v_template, batch_size=1)
    table_params = params2torch(seq_data["table"]["params"])

    # assume table is static
    table_params["transl"] = table_params["transl"][[0]]
    table_params["global_orient"] = table_params["global_orient"][[0]]
    table_verts = table_model(**table_params).vertices
    table_offset = torch.tensor([0.0, 0.0, 0.0])
    table_verts += table_offset
    table_corner_locs = table_verts[0, TABLE_CORNER_VERT_IDS_SIDE1]
    v1 = table_corner_locs[1] - table_corner_locs[0]
    v2 = table_corner_locs[2] - table_corner_locs[1]
    v3 = table_corner_locs[3] - table_corner_locs[2]
    v4 = table_corner_locs[0] - table_corner_locs[3]
    assert table_corner_locs.shape == (
        4,
        3,
    ), "Table corners must be 4 points in 3D space"
    if torch.cross(v1, v2)[2] < 0:
        table_corner_locs = table_verts[0, TABLE_CORNER_VERT_IDS_SIDE2]
    v1 = table_corner_locs[1] - table_corner_locs[0]
    v2 = table_corner_locs[2] - table_corner_locs[1]
    v3 = table_corner_locs[3] - table_corner_locs[2]
    v4 = table_corner_locs[0] - table_corner_locs[3]
    assert torch.cross(v1, v2)[2] > 0
    assert torch.cross(v2, v3)[2] > 0
    assert torch.cross(v3, v4)[2] > 0
    assert torch.cross(v4, v1)[2] > 0
    return table_faces, table_verts, table_corner_locs
