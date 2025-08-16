import glob
import numpy as np
import os
import smplx
import trimesh
import bpy
import torch
import pickle

from blender_utils.visualize_mesh_figure_blender import (
    animate_mesh,
    create_new_blender_object_with_mesh,
)
from blender_utils.general_blender_utils import blend_scp_and_run
from datasets.grab.grab_utils import (
    grab_seq_path_to_seq_id,
    grab_seq_path_to_unique_name,
    params2torch,
    parse_npz,
    reduce_seq_data,
    simplify_trimesh,
    to_cpu,
)
from resource_paths import GRAB_ORIG_DATA_PATH, GRAB_DATA_PATH
from general_utils import TMP_DIR, create_new_dir
from objects_fk import ObjectModel
from datasets.smpldata import SMPL_MODELS_PATH


HAND_DATA_SOURCE_MODES = [
    "body_hand_pose_24",
    "body_fullpose_165",
    "hand_hand_pose_24",
    "hand_fullpose_45",
]

"""
use_betas=False, use_betas_zero=None,  hand_data_source="hand_fullpose_45", gender=None works
use_betas=True , use_betas_zero=False, hand_data_source="hand_fullpose_45", gender=None works
use_betas=True , use_betas_zero=True, hand_data_source="hand_fullpose_45",  gender='neutral' doesn't

"""


def animate_grab_fullbody(
    animation_path: str,
    save_path: str = None,
    reset_blender: bool = True,
    tgt_fps: int = 20,
    start_frame: int = None,
    end_frame: int = None,
    use_betas=False,
    use_betas_zero=True,
    hand_data_source="hand_fullpose_45",
    gender_to_use=None,
    unq_id="",
):
    """
    gender: if None, use the gender of the subject in the animation
    use_betas: if True, use the betas of the subject in the animation
    use_betas_zero: if True, use zero betas
    """
    seq_data = parse_npz(animation_path)
    seq_data = reduce_seq_data(seq_data, tgt_fps, start_frame, end_frame)
    # seq_data = torchify_numpy_dict_recursive(seq_data)
    n_comps = seq_data["n_comps"]
    sbj_gender = seq_data["gender"]
    gender_to_use = gender_to_use or sbj_gender

    T = seq_data["n_frames"]
    sbj_mesh_path = os.path.join(
        GRAB_DATA_PATH, sbj_gender, os.path.basename(seq_data["body"]["vtemp"])
    )
    sbj_betas_path = sbj_mesh_path.replace(".ply", "_betas.npy")
    sbj_betas = np.load(sbj_betas_path)

    obj_mesh_path = os.path.join(
        GRAB_DATA_PATH,
        os.path.relpath(
            seq_data["object"]["object_mesh"],
            os.path.dirname(os.path.dirname(seq_data["object"]["object_mesh"])),
        ),
    )
    table_mesh_path = os.path.join(
        GRAB_DATA_PATH,
        os.path.relpath(
            seq_data["table"]["table_mesh"],
            os.path.dirname(os.path.dirname(seq_data["table"]["table_mesh"])),
        ),
    )

    if reset_blender:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    if use_betas:
        smplx_model_path = os.path.join(
            SMPL_MODELS_PATH, f"smplx/SMPLX_{gender_to_use.upper()}.pkl"
        )
        with open(smplx_model_path, "rb") as f:
            d = pickle.load(f, encoding="latin")
        sbj_mesh_faces = d["f"]
        sbj_mesh_vertices = d["v_template"]
    else:
        sbj_mesh = trimesh.load(sbj_mesh_path)
        sbj_mesh_faces = np.array(sbj_mesh.faces)
        sbj_mesh_vertices = np.array(sbj_mesh.vertices)

    sbj_params = params2torch(seq_data["body"]["params"])

    assert n_comps == 24, f"n_comps={n_comps} must be 24"
    if hand_data_source == "body_hand_pose_24":
        left_hand_pose = seq_data["body"]["params"]["left_hand_pose"]
        right_hand_pose = seq_data["body"]["params"]["right_hand_pose"]
        use_pca = True
    elif hand_data_source == "body_fullpose_165":
        left_hand_pose = seq_data["body"]["params"]["fullpose"][
            :, 25 * 3 : 40 * 3
        ]  # double check indices!
        right_hand_pose = seq_data["body"]["params"]["fullpose"][
            :, 40 * 3 : 55 * 3
        ]  # double check indices!
        use_pca = False
    elif hand_data_source == "hand_hand_pose_24":
        left_hand_pose = seq_data["lhand"]["params"]["hand_pose"]
        right_hand_pose = seq_data["rhand"]["params"]["hand_pose"]
        use_pca = True
    elif hand_data_source == "hand_fullpose_45":
        left_hand_pose = seq_data["lhand"]["params"]["fullpose"]
        right_hand_pose = seq_data["rhand"]["params"]["fullpose"]
        use_pca = False
    else:
        raise ValueError(
            f"hand_data_source={hand_data_source} must be one of {HAND_DATA_SOURCE_MODES}"
        )
    left_hand_pose = torch.from_numpy(left_hand_pose).float()
    right_hand_pose = torch.from_numpy(right_hand_pose).float()

    # human
    sbj_model: smplx.SMPLX = smplx.create(
        model_path=SMPL_MODELS_PATH,
        model_type="smplx",
        gender=gender_to_use or sbj_gender,
        num_pca_comps=n_comps if use_pca else None,
        v_template=sbj_mesh_vertices if not use_betas else None,
        batch_size=T,
        use_pca=use_pca,
        flat_hand_mean=True,
        # create_expression=True,
        # create_jaw_pose=True,
        # create_leye_pose=True,
        # create_reye_pose=True,
    )

    sbj_betas = np.tile(sbj_betas, (T, 1))
    sbj_betas = torch.from_numpy(sbj_betas).float()
    if use_betas_zero:
        # sbj_betas = torch.zeros_like(sbj_betas)
        sbj_betas = None
    smplx_output = sbj_model(
        betas=sbj_betas,
        global_orient=sbj_params["global_orient"],
        body_pose=sbj_params["body_pose"],
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        transl=sbj_params["transl"],
        # expression=sbj_params['expression'],
        # jaw_pose=sbj_params['jaw_pose'],
        # leye_pose=sbj_params['leye_pose'],
        # reye_pose=sbj_params['reye_pose'],
    )

    sbj_verts_anim = to_cpu(smplx_output.vertices)
    animate_mesh(f"SMPLX{unq_id}", sbj_mesh_faces, sbj_verts_anim)

    # object
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_mesh = simplify_trimesh(obj_mesh, tgt_faces=1500)
    obj_faces = np.array(obj_mesh.faces)
    obj_verts = np.array(obj_mesh.vertices)
    obj_model = ObjectModel(v_template=obj_verts, batch_size=T)
    obj_params = params2torch(seq_data["object"]["params"])
    obj_verts_anim = to_cpu(obj_model(**obj_params).vertices)
    animate_mesh(f"Object{unq_id}", obj_faces, obj_verts_anim)

    # table
    table_mesh = trimesh.load(table_mesh_path)
    table_faces = np.array(table_mesh.faces)
    table_verts = np.array(table_mesh.vertices)
    table_model = ObjectModel(v_template=table_verts, batch_size=T)
    table_parms = params2torch(seq_data["table"]["params"])
    table_verts_located = to_cpu(table_model(**table_parms).vertices)
    create_new_blender_object_with_mesh(
        f"Table{unq_id}", table_verts_located[0], table_faces.tolist()
    )

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


def main_test_sources():
    animation_path = (
        "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB/s2/banana_eat_1.npz"
    )
    animation_path = (
        "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB/s2/teapot_lift.npz"
    )
    animation_path = "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB/s1/alarmclock_offhand_1.npz"
    # out_dir = "/home/dcor/roeyron/tmp/grab"
    for hand_data_source in HAND_DATA_SOURCE_MODES:
        save_path = os.path.join(
            TMP_DIR,
            os.path.basename(animation_path).replace(
                ".npz", f"_{hand_data_source}.blend"
            ),
        )
        if os.path.exists(save_path):
            continue
        animate_grab_fullbody(
            animation_path,
            save_path,
            tgt_fps=5,
            start_frame=40,
            hand_data_source=hand_data_source,
        )
        print(f"Saved to {save_path}")


def main_test_setups():
    animation_path = (
        "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB/s1/teapot_pour_1.npz"
    )

    configs = [
        {
            "use_betas": False,
            "use_betas_zero": None,
            "gender_to_use": None,
        },  # use v_template
        # {'use_betas': True, 'use_betas_zero': False, 'gender_to_use': None},  # use fitted betas and same gender
        # {'use_betas': True, 'use_betas_zero': True, 'gender_to_use': 'neutral'},  # use zero betas, neutral gender
    ]
    save_dir = os.path.join(TMP_DIR, "grab_neutral_retargeted")
    create_new_dir(save_dir)
    for config in configs:
        print(f"Running config: {config}")
        # animation_path = "/home/dcor/roeyron/tmp/grab_neutral_retargeted/s1/teapot_pour_1.npz"
        use_betas = config["use_betas"]
        use_betas_zero = config["use_betas_zero"]
        gender_to_use = config["gender_to_use"]
        conf_str = f"use_betas_{use_betas}__use_betas_zero_{use_betas_zero}__gender_{gender_to_use}"
        fname = os.path.basename(animation_path).replace(".npz", f"__{conf_str}.blend")
        save_path = os.path.join(save_dir, fname)
        if os.path.exists(save_path):
            continue
        animate_grab_fullbody(
            animation_path,
            save_path,
            tgt_fps=5,
            start_frame=0,
            use_betas=use_betas,
            use_betas_zero=use_betas_zero,
            gender_to_use=gender_to_use,
        )
    print(
        f"scp -r roeyron@c-005.cs.tau.ac.il:{save_dir} ~/Downloads/{os.path.basename(save_dir)}"
    )


# def compare_orig_and_retargeted():
#     # animation_path_orig = "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB/s1/airplane_fly_1.npz"
#     # animation_path_retg = "/home/dcor/roeyron/tmp/grab_neutral_retargeted/s1/airplane_fly_1.npz"
#     save_dir = os.path.join(TMP_DIR, "grab_neutral_retargeted_comparison")
#     create_new_dir(save_dir)
#     retg_paths = glob.glob("/home/dcor/roeyron/tmp/grab_neutral_retargeted/**/*.npz")
#     retg_paths = retg_paths[::-1]
#     for retg_seq_path in retg_paths:
#         seq_id = grab_seq_path_to_seq_id(retg_seq_path)
#         print(f"Processing {seq_id}")
#         orig_seq_path = os.path.join(GRAB_DATA_PATH, f"{seq_id}.npz")
#         seq_str = grab_seq_path_to_unique_name(orig_seq_path)
#         save_path = os.path.join(save_dir, f"{seq_str}_comparison.blend")
#         animate_grab_fullbody(orig_seq_path, None, tgt_fps=5, use_betas=False, unq_id="Orig")
#         animate_grab_fullbody(orig_seq_path, None, tgt_fps=5, reset_blender=False, use_betas=True, use_betas_zero=True, gender_to_use='neutral', unq_id="OrigNeutral")
#         animate_grab_fullbody(retg_seq_path, save_path, tgt_fps=5, reset_blender=False, use_betas=True, use_betas_zero=True, gender_to_use='neutral', unq_id="RetargetedNeutral")
#     print(f"scp -r roeyron@c-005.cs.tau.ac.il:{save_dir} ~/Downloads/{os.path.basename(save_dir)}")


def compare_orig_and_retargeted():
    save_dir = os.path.join(TMP_DIR, "grab_retargeted_animations")
    create_new_dir(save_dir)
    seq_fnames = glob.glob("**/*.npz", root_dir=GRAB_DATA_PATH)
    seq_fnames = np.random.choice(seq_fnames, size=10, replace=False)
    for seq_fname in seq_fnames:
        print(f"Processing {seq_fname}")
        rtgt_seq_path = os.path.join(GRAB_DATA_PATH, seq_fname)
        orig_seq_path = os.path.join(GRAB_ORIG_DATA_PATH, seq_fname)
        seq_str = grab_seq_path_to_unique_name(orig_seq_path)
        save_path = os.path.join(save_dir, f"{seq_str}_comparison.blend")
        animate_grab_fullbody(
            orig_seq_path, None, tgt_fps=5, use_betas=False, unq_id="Orig"
        )
        animate_grab_fullbody(
            orig_seq_path,
            None,
            tgt_fps=5,
            reset_blender=False,
            use_betas=True,
            use_betas_zero=True,
            gender_to_use="neutral",
            unq_id="OrigNeutral",
        )
        animate_grab_fullbody(
            rtgt_seq_path,
            save_path,
            tgt_fps=5,
            reset_blender=False,
            use_betas=True,
            use_betas_zero=True,
            gender_to_use="neutral",
            unq_id="RetargetedNeutral",
        )
    print(
        f"scp -r roeyron@c-005.cs.tau.ac.il:{save_dir} ~/Downloads/{os.path.basename(save_dir)}"
    )


def main():
    animation_path = "/home/dcor/roeyron/trumans_utils/DATASETS/DATA_GRAB_RETARGETED/s2/airplane_fly_1.npz"
    save_path = "/home/dcor/roeyron/tmp/s2_airplane_fly_neutral_retargeted.blend"
    animate_grab_fullbody(
        animation_path,
        save_path,
        tgt_fps=5,
        use_betas=True,
        use_betas_zero=True,
        gender_to_use="neutral",
        hand_data_source="hand_fullpose_45",
    )


if __name__ == "__main__":
    main()
    # main_test_sources()
    # compare_orig_and_retargeted()
    # compare_orig_and_retargeted()


"""
datasets/grab/animate_grab_fullbody.py
"""
