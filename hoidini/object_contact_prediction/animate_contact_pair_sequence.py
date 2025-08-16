import os
import pickle
from typing import List, Optional, Tuple
import bpy
import numpy as np
import torch
import trimesh

from hoidini.blender_utils.general_blender_utils import (
    CollectionManager,
    save_blender_file,
)
from hoidini.blender_utils.visualize_mesh_figure_blender import (
    animate_mesh,
    animate_rigid_mesh,
    create_new_blender_object_with_mesh,
)
from hoidini.blender_utils.general_blender_utils import blend_scp_and_run
from hoidini.datasets.grab.animate_grab_contacts import add_contact_spheres
from hoidini.datasets.grab.animate_grab_hands import place_table
from hoidini.datasets.grab.grab_utils import (
    grab_object_name_to_path,
    load_mesh,
    params2torch,
    parse_npz,
    simplify_trimesh,
    to_cpu,
)
from hoidini.datasets.smpldata import SMPL_MODELS_PATH
from hoidini.general_utils import TMP_DIR, get_distinguishable_rgba_colors
from hoidini.object_contact_prediction.cpdm_dataset import (
    ContactPairsDataset,
    ContactPairsSequence,
    get_contact_pairs_dataloader,
    load_contact_pairs_sequence,
)
from hoidini.objects_fk import ObjectModel
from hoidini.resource_paths import GRAB_DATA_PATH
from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info
import hoidini.smplx as smplx


def animate_contact_points(
    contact_points_lst: List[List[np.ndarray]],
    unq_id: str,
    color: tuple | List[tuple] = (1, 0, 0, 0.7),
    sphere_size: float = 0.01,
):
    """
    contact_points[frame][anchor_ind] = (bool, (x, y, z))
        frame in [0, len(contact_points_lst) - 1]
        anchor_ind in [0, n_anchors - 1]
    """
    n_anchors = len(contact_points_lst[0])
    sphere_contacts = add_contact_spheres(
        n_anchors,
        f"Contacts{unq_id}",
        f"contact{unq_id}",
        color=color,
        radius=sphere_size,
    )
    for frame in range(len(contact_points_lst)):
        contact_locs = contact_points_lst[frame]
        for ind, (bit, loc) in enumerate(contact_locs):
            sphere_contact = sphere_contacts[ind]
            if bit:
                sphere_contact.location = loc
                sphere_contact.hide_viewport = False
                sphere_contact.keyframe_insert(data_path="location", frame=frame)
                sphere_contact.keyframe_insert(data_path="hide_viewport", frame=frame)
            else:
                sphere_contact.hide_viewport = True
                sphere_contact.keyframe_insert(data_path="hide_viewport", frame=frame)


def add_offset(a, offset):
    """
    a: torch.Tensor or np.ndarray
    offset: torch.Tensor
    """
    if isinstance(a, torch.Tensor):
        offset = offset.to(a.device)
        return a + offset.reshape((1,) * (a.ndim - 1) + (3,))
    elif isinstance(a, np.ndarray):
        offset_np = offset.cpu().numpy() if isinstance(offset, torch.Tensor) else offset
        return a + offset_np.reshape((1,) * (a.ndim - 1) + (3,))


def animate_contact_pair_sequence(
    cps: ContactPairsSequence,
    object_name: str,
    save_path: str = None,
    unq_id: str = "",
    reset_blender: bool = True,
    offset_all: torch.Tensor | None = None,
    offset_all_but_6dof: torch.Tensor | None = None,
    contact_threshold: float = 0.5,
    use_multicolor_anchors: bool = True,
    meshes_color: Optional[Tuple[float, float, float, float]] = None,
    grab_seq_path_for_table: str = None,
    sphere_size: float = 0.01,
):

    if reset_blender:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    if offset_all is not None:
        offset_all_but_6dof = offset_all
        offset_6dof = offset_all
    elif offset_all_but_6dof is not None:
        offset_6dof = torch.tensor([0, 0, 0])
    elif offset_all is None and offset_all_but_6dof is None:
        offset_all_but_6dof = torch.tensor([0, 0, 0])
        offset_6dof = torch.tensor([0, 0, 0])

    if use_multicolor_anchors:
        colors_all = get_distinguishable_rgba_colors(cps.local_object_points.shape[1])
    else:
        colors_all = [(1, 0, 0, 0.9)] * cps.local_object_points.shape[1]

    ############################################################
    # 0) Place the table
    ############################################################
    if grab_seq_path_for_table:
        seq_data = parse_npz(
            grab_seq_path_for_table
        )  # no need to reduce fps, since it's static
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
        table_verts_located = add_offset(table_verts_located, offset_all_but_6dof)
        create_new_blender_object_with_mesh(
            f"Table{unq_id}",
            table_verts_located[0],
            table_faces.tolist(),
            color=meshes_color,
        )

    ############################################################
    # 1) Place the static object mesh
    ############################################################
    obj_verts, obj_faces = load_mesh(object_name, n_simplify_faces=3000)
    # obj_faces = np.array(obj_faces)
    # obj_verts = np.array(obj_verts)
    animate_mesh(
        f"Static_Obj{unq_id}",
        obj_faces,
        add_offset(obj_verts[None], offset_all_but_6dof),
        color=meshes_color,
    )
    # animate_rigid_mesh(f"Static_Obj{unq_id}", obj_faces, add_offset(obj_verts, offset_all_but_6dof), color=meshes_color)

    ############################################################
    # 2) Visualize local object contact predictions
    ############################################################
    local_contact_points_lst = []
    for frame in range(len(cps)):
        locs = cps.local_object_points[frame].detach().cpu().numpy()
        locs = add_offset(locs, offset_all_but_6dof)
        locs = list(locs)
        bits = list((cps.contacts[frame] > contact_threshold).detach().cpu().numpy())
        local_contact_points = list(zip(bits, locs))
        local_contact_points_lst.append(local_contact_points)
    animate_contact_points(
        local_contact_points_lst,
        f"Local{unq_id}",
        color=colors_all,
        sphere_size=sphere_size,
    )

    ############################################################
    # 3) Move the object according to translations and rotations
    ############################################################
    obj_model = ObjectModel(v_template=obj_verts, batch_size=len(cps))
    obj_params = {"global_orient": cps.object_poses, "transl": cps.object_trans}
    obj_verts_anim = add_offset(obj_model(**obj_params).vertices, offset_6dof)
    obj_verts_anim = obj_verts_anim.detach().cpu().numpy()
    animate_mesh(f"Dynamic_Obj{unq_id}", obj_faces, obj_verts_anim, color=meshes_color)
    # animate_rigid_mesh(f"Dynamic_Obj{unq_id}", obj_faces, obj_verts, obj_params['global_orient'], obj_params['transl'], color=meshes_color)

    ############################################################
    # 4) Show contact of static hands
    ############################################################
    for hand in ["left", "right"]:
        hand_mano_model_path = os.path.join(
            SMPL_MODELS_PATH, f"mano/MANO_{hand.upper()}.pkl"
        )
        with open(hand_mano_model_path, "rb") as f:
            d = pickle.load(f, encoding="latin")
        mesh_faces = d["f"]
        hand_model: smplx.MANO = smplx.create(
            model_path=SMPL_MODELS_PATH,
            is_rhand=hand == "right",
            model_type="mano",
            gender="neutral",
            flat_hand_mean=True,
            use_pca=False,
            batch_size=1,
        )
        hand_model.SHAPE_SPACE_DIM = 10  # to disable warning message
        mano_output: smplx.MANO.forward = hand_model(
            global_orient=(
                torch.tensor([[0, -np.pi / 2, 0]])
                if hand == "left"
                else torch.tensor([[0, np.pi / 2, 0]])
            ),
            hand_pose=torch.zeros((1, 45)),
            transl=(
                torch.tensor([[0.4, 0, 0]])
                if hand == "left"
                else torch.tensor([[-0.4, 0, 0]])
            ),
        )

        hand_verts_anim = (
            add_offset(mano_output.vertices, offset_all_but_6dof).detach().cpu().numpy()
        )
        animate_mesh(
            f"Mano{hand}_{unq_id}", mesh_faces, hand_verts_anim, color=meshes_color
        )

        anchor_inds_R2hands = get_contact_anchors_info()[0]
        anchor_locs = hand_verts_anim[0, anchor_inds_R2hands]

        n_anchors = cps.local_object_points.shape[1] // 2
        hand_anchor_inds = (
            torch.arange(n_anchors)
            if hand == "left"
            else torch.arange(n_anchors) + n_anchors
        )
        contact_points_lst = []
        for frame in range(len(cps)):
            bits = list(
                (cps.contacts[frame, hand_anchor_inds] > contact_threshold)
                .detach()
                .cpu()
                .numpy()
            )
            locs = anchor_locs
            contact_points = list(zip(bits, locs))
            contact_points_lst.append(contact_points)
        if use_multicolor_anchors:
            colors_hand = (
                colors_all[:n_anchors] if hand == "left" else colors_all[n_anchors:]
            )
        else:
            colors_hand = [(0, 1, 0, 0.9)] * n_anchors
        animate_contact_points(
            contact_points_lst,
            f"{hand}Contact{unq_id}",
            color=colors_hand,
            sphere_size=sphere_size,
        )

    if save_path is not None:
        save_blender_file(save_path)
        blend_scp_and_run(save_path)


def main_from_contact_pairs_sequence():
    grab_seq_path = os.path.join(GRAB_DATA_PATH, "s2/teapot_pour_1.npz")
    save_path = os.path.join(TMP_DIR, "contact_pair_vis.blend")
    contact_pairs_sequence, object_name = load_contact_pairs_sequence(grab_seq_path)
    animate_contact_pair_sequence(contact_pairs_sequence, object_name, save_path)


def main():
    grab_dataset_path = GRAB_DATA_PATH
    use_normalizer = True
    batch_size = 2
    lim = 2
    seed = 0
    context_len = 100
    pred_len = 10
    what_to_animate = "context"  # "pred" or "context"

    dataloader = get_contact_pairs_dataloader(
        grab_dataset_path,
        batch_size=batch_size,
        context_len=context_len,
        pred_len=pred_len,
        experiment_dir="/home/dcor/roeyron/trumans_utils/src/EXPERIMENTS/cpdm_debug",
        is_training=True,
        lim=lim,
        use_normalizer=use_normalizer,
        seed=seed,
    )
    motion, cond = next(iter(dataloader))
    if what_to_animate == "pred":
        features = motion
    elif what_to_animate == "context":
        features = cond["y"]["prefix"]

    dataset: ContactPairsDataset = dataloader.dataset
    feature_processor = dataset.feature_processor

    save_path = os.path.join(TMP_DIR, "contact_pair_vis_motion.blend")

    # 1.a) visualize from dataloader motion ("pred")
    i = 1
    features = features.squeeze(2).permute(
        0, 2, 1
    )  # (bs, n_features, 1, seq_len) --> (bs, seq_len, n_features)
    features_i = features[i]  # (seq_len, n_features)
    object_name = cond["y"]["metadata"][i]["object_name"]
    if use_normalizer:
        features_i = dataset.normalizer.denormalize(features_i)
    cps_motion = feature_processor.decode_features(features_i)
    animate_contact_pair_sequence(
        cps_motion,
        object_name,
        save_path=None,
        unq_id=f"Dataloader{i}",
        offset_all=torch.tensor([0, 0, 0.25]),
    )

    # 1.b) Visualize prefix ("context")
    # cps_cond = feature_processor.decode_features(cond['y']['prefix'])
    # animate_contact_pair_sequence(cps_cond, cond['object_name'], reset_blender=False,
    #                               save_path=os.path.join(TMP_DIR, "contact_pair_vis_cond.blend"))

    # 2.a) Visualize from contact pairs sequence ("pred")
    grab_seq_path = cond["y"]["metadata"][i]["grab_seq_path"]
    start_frame, end_frame = cond["y"]["metadata"][i]["range"]
    if what_to_animate == "pred":
        start_frame = start_frame + context_len
    elif what_to_animate == "context":
        end_frame = end_frame - pred_len
    contact_pairs_sequence, object_name = load_contact_pairs_sequence(
        grab_seq_path, start_frame=start_frame, end_frame=end_frame
    )
    animate_contact_pair_sequence(
        contact_pairs_sequence,
        object_name,
        save_path=None,
        reset_blender=False,
        unq_id=f"ContactPairsSequence{i}",
    )

    #################
    # Organize Collections
    #################
    substr_map = {
        "Dataloader": "Dataloader",
        "ContactPairsSequence": "ContactPairsSequence",
    }
    CollectionManager.organize_collections(substr_map)

    save_blender_file(save_path)
    blend_scp_and_run(save_path)


if __name__ == "__main__":
    # main_from_contact_pairs_sequence()
    main()
