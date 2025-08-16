import numpy as np
import os
import smplx
import trimesh
import bpy
import torch
import pickle
from torch import Tensor

from hoidini.blender_utils.visualize_mesh_figure_blender import (
    animate_mesh,
    create_new_blender_object_with_mesh,
)
from hoidini.blender_utils.general_blender_utils import blend_scp_and_run
from hoidini.datasets.grab.grab_utils import (
    grab_object_name_to_path,
    grab_seq_data_to_object_path,
    params2torch,
    parse_npz,
    simplify_trimesh,
    to_cpu,
    reduce_seq_data,
    translate_seq_data,
)
from hoidini.objects_fk import ObjectModel
from hoidini.datasets.smpldata import SMPL_MODELS_PATH


def place_table(seq_data, blender_suffix="", table_mesh_path=None, color=None):
    if table_mesh_path is None:
        table_mesh_path = grab_object_name_to_path("table")
    table_mesh = trimesh.load(table_mesh_path)
    table_faces = np.array(table_mesh.faces)
    table_verts = np.array(table_mesh.vertices)
    table_model = ObjectModel(v_template=table_verts, batch_size=1)
    table_params = params2torch(seq_data["table"]["params"])
    # assume table is static
    table_params["transl"] = table_params["transl"][[0]]
    table_params["global_orient"] = table_params["global_orient"][[0]]
    table_verts_located = to_cpu(table_model(**table_params).vertices)
    create_new_blender_object_with_mesh(
        f"Table{blender_suffix}",
        table_verts_located[0],
        table_faces.tolist(),
        color=color,
    )


def animate_grab_object_and_hands(
    grab_anim_path: str,
    save_path: str = None,
    reset_blender: bool = True,
    tgt_fps: int = 20,
    start_frame: int = None,
    end_frame: int = None,
    translation: np.ndarray | Tensor = None,
    blender_suffix: str = "",
):
    grab_data_path = os.path.dirname(os.path.dirname(grab_anim_path))
    seq_data = parse_npz(grab_anim_path)
    seq_data = reduce_seq_data(seq_data, tgt_fps, start_frame, end_frame)
    if translation is not None:
        seq_data = translate_seq_data(seq_data, translation)

    n_comps = seq_data["n_comps"]
    gender = seq_data["gender"]
    T = seq_data["n_frames"]

    lhand_mesh_path = os.path.join(
        grab_data_path, seq_data["gender"], os.path.basename(seq_data["lhand"]["vtemp"])
    )
    rhand_mesh_path = os.path.join(
        grab_data_path, seq_data["gender"], os.path.basename(seq_data["rhand"]["vtemp"])
    )
    lhand_betas_path = lhand_mesh_path.replace(".ply", "_betas.npy")
    rhand_betas_path = rhand_mesh_path.replace(".ply", "_betas.npy")

    hand_betas = {
        "left": np.load(lhand_betas_path),
        "right": np.load(rhand_betas_path),
    }

    hand_keys = {"left": "lhand", "right": "rhand"}

    hand_mesh_paths = {
        "left": lhand_mesh_path,
        "right": rhand_mesh_path,
    }

    use_betas = True
    use_betas_zero = True

    obj_mesh_path = grab_seq_data_to_object_path(seq_data)
    table_mesh_path = grab_object_name_to_path("table")

    if reset_blender:
        bpy.ops.wm.read_factory_settings(use_empty=True)

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
        hand_verts_anim = to_cpu(mano_output.vertices)
        animate_mesh(f"Mano_{hand}{blender_suffix}", mesh_faces, hand_verts_anim)

    # object
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_mesh = simplify_trimesh(obj_mesh, tgt_faces=1500)
    obj_faces = np.array(obj_mesh.faces)
    obj_verts = np.array(obj_mesh.vertices)
    obj_model = ObjectModel(v_template=obj_verts, batch_size=T)
    obj_params = params2torch(seq_data["object"]["params"])
    obj_verts_anim = to_cpu(obj_model(**obj_params).vertices)
    animate_mesh(f"Object{blender_suffix}", obj_faces, obj_verts_anim)

    # table
    place_table(seq_data, blender_suffix, table_mesh_path)

    scene = bpy.context.scene
    scene.frame_start = 0
    if end_frame is not None:
        scene.frame_end = end_frame
    scene.render.fps = tgt_fps

    if save_path is not None:
        if os.path.exists(save_path):
            os.remove(save_path)
        bpy.ops.wm.save_mainfile(filepath=save_path)
        blend_scp_and_run(save_path)


def main():
    animation_path = (
        "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB/s2/teapot_lift.npz"
    )
    save_path = "/home/dcor/roeyron/tmp/teapot_lift_only_hands.blend"
    animate_grab_object_and_hands(animation_path, save_path, tgt_fps=5, end_frame=50)


if __name__ == "__main__":
    main()
