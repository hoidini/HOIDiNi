from __future__ import annotations
from enum import Enum
from typing import Optional
import bpy
import torch
import os
import time
import seaborn as sns

from hoidini.datasets.grab.grab_object_records import ContactRecord
from hoidini.object_contact_prediction.animate_contact_pair_sequence import (
    animate_contact_pair_sequence,
)
from hoidini.object_contact_prediction.cpdm_dataset import ContactPairsSequence
from hoidini.blender_utils.visualize_mesh_figure_blender import (
    animate_mesh,
    animate_rigid_mesh,
)
from hoidini.blender_utils.visualize_stick_figure_blender import (
    save_blender_file,
    visualize_motions,
)
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.datasets.grab.animate_grab_contacts import (
    animate_contact_record,
    animate_predicted_contacts,
)
from hoidini.datasets.grab.animate_grab_hands import (
    animate_grab_object_and_hands,
    place_table,
)
from hoidini.datasets.grab.grab_utils import (
    grab_object_name_to_path,
    load_mesh,
    parse_npz,
)
from hoidini.datasets.smpldata import SmplData, SmplModelsFK
from hoidini.objects_fk import ObjectModel
from hoidini.general_utils import TMP_DIR
from hoidini.blender_utils.general_blender_utils import (
    CollectionManager,
    blend_scp_and_run,
)


class AnimationSetup(Enum):
    NO_MESH = 0  # All frames, no mesh visualization
    MESH_PARTIAL = 1  # Every 5th frame with mesh visualization
    MESH_ALL = 2  # All frames with mesh visualization (slow)

    @classmethod
    def get_animation_params(cls, anim_setup: AnimationSetup | str):
        if isinstance(anim_setup, str):
            anim_setup = AnimationSetup[anim_setup]
        if anim_setup == AnimationSetup.NO_MESH:
            skip = 1
            frame_lim = None
            vis_mesh = False
        elif anim_setup == AnimationSetup.MESH_PARTIAL:
            skip = 5
            frame_lim = None
            vis_mesh = True
        elif anim_setup == AnimationSetup.MESH_ALL:
            skip = 1
            frame_lim = None
            vis_mesh = True
        else:
            raise ValueError(
                f"anim_setup must be an AnimationSetup enum value, got {anim_setup}"
            )

        return skip, frame_lim, vis_mesh


def reduce_sequences(sequences, skip, frame_lim):
    if sequences is not None:
        return [e[::skip][:frame_lim] for e in sequences]


def visualize_hoi_animation(
    smpldata_lst: list[SmplData],
    *,
    object_path_or_name: str | None = None,
    grab_seq_path: str | None = None,
    start_frame: int | None = None,
    text: str = "",
    translation_obj=None,
    anim_setup=AnimationSetup.NO_MESH,
    contact_record: Optional[ContactRecord] = None,
    save_path=None,
    mat_lst=None,
    model_id_str: str = "",
    contact_pairs_seq: Optional[
        ContactPairsSequence | list[ContactPairsSequence]
    ] = None,
    n_simplify_object: int = 3000,
    text_config: str = None,
    smplx_cancel_offset: bool = True,
    reset_blender: bool = True,
    unq_id: str = "",
    save=True,
    colors_dict: dict[str, tuple] = None,
    table_grab_seq_path: str | None = None,
    visualize_from_joints_as_well: bool = False,
    visualize_predicted_6dof: bool = True,
):
    """
    Visualize HOI animation
    """
    tic = time.time()
    n_frames = len(smpldata_lst[0])
    colors_dict = colors_dict or {}
    if isinstance(anim_setup, str):
        anim_setup = AnimationSetup[anim_setup]

    if reset_blender:
        bpy.ops.wm.read_factory_settings(use_empty=True)

    # Get animation parameters
    skip, frame_lim, vis_mesh = AnimationSetup.get_animation_params(anim_setup)

    if table_grab_seq_path is not None:
        seq_data = parse_npz(table_grab_seq_path)
        place_table(seq_data, blender_suffix=f"{unq_id}Table")

    #################
    # Visualize human
    #################
    smpl_fk = SmplModelsFK.create("smplx", n_frames, device=dist_util.dev())
    joint_lst = []
    verts_lst = []
    for i, smpldata in enumerate(smpldata_lst):
        with torch.no_grad():
            # Understand the cancel offset parameter!!!
            smpl_output = smpl_fk.smpldata_to_smpl_output(
                smpldata.to(dist_util.dev()), cancel_offset=smplx_cancel_offset
            )
        joints_from_pose = smpl_output.joints.detach().cpu()
        verts_from_pose = smpl_output.vertices.detach().cpu()
        smpldata.joints_from_poses = (
            joints_from_pose  # <--- set joints_from_pose (fk output)
        )
        joint_lst.append(joints_from_pose)
        verts_lst.append(verts_from_pose)

    joint_lst = reduce_sequences(joint_lst, skip, frame_lim)
    verts_lst = reduce_sequences(verts_lst, skip, frame_lim)
    # visualize human
    visualize_motions(
        joint_lst,
        save_path=None,
        verts_anim_lst=verts_lst if vis_mesh else None,
        colors=(
            None if "stick_figure" not in colors_dict else [colors_dict["stick_figure"]]
        ),
        mat_lst=mat_lst,
        unq_name=f"{unq_id}stick",
        reset_blender=False,
    )
    CollectionManager.organize_collections(f"{unq_id}stick")

    # visualize human from joints
    if visualize_from_joints_as_well:
        visualize_motions(
            [sd.joints for sd in smpldata_lst],
            unq_name=f"{unq_id}stickFromJoints",
            save_path=None,
            verts_anim_lst=None if not vis_mesh else verts_lst,
            colors=None,
            reset_blender=False,
        )
        CollectionManager.organize_collections(f"{unq_id}stickFromJoints")
    #################
    # Visualize predicted object 6Dof
    #################
    if visualize_predicted_6dof and object_path_or_name is not None:
        if os.path.exists(object_path_or_name):
            obj_mesh_path = object_path_or_name
        else:
            obj_mesh_path = grab_object_name_to_path(object_path_or_name)
        obj_verts, obj_faces = load_mesh(
            obj_mesh_path, n_simplify_faces=n_simplify_object
        )
        for i, smpldata in enumerate(smpldata_lst):

            obj_model = ObjectModel(v_template=obj_verts, batch_size=len(smpldata))

            joints_from_pose = joint_lst[i]
            colors = list(sns.color_palette("Set2"))
            for j, extract_obj_loc_from in enumerate(["body", "lhand", "rhand"]):
                if extract_obj_loc_from == "body":
                    poses_obj = smpldata.poses_obj
                    transl_obj = smpldata.trans_obj
                elif extract_obj_loc_from == "lhand":
                    poses_obj = smpldata.poses_obj_from_lhand
                    transl_obj = smpldata.trans_obj_from_lhand
                elif extract_obj_loc_from == "rhand":
                    poses_obj = smpldata.poses_obj_from_rhand
                    transl_obj = smpldata.trans_obj_from_rhand
                else:
                    raise ValueError(
                        f"extract_obj_loc_from must be one of ['body', 'left_hand', 'right_hand'], got {extract_obj_loc_from}"
                    )
                if transl_obj is None or poses_obj is None:
                    continue

                obj_params = {
                    "transl": transl_obj.detach().cpu(),
                    "global_orient": poses_obj.detach().cpu(),
                }
                obj_verts_anim = obj_model(**obj_params).vertices.numpy()
                obj_verts_anim = [e[::skip][:frame_lim] for e in obj_verts_anim]
                animate_mesh(
                    f"{unq_id}6DoF{i}_{extract_obj_loc_from}",
                    obj_faces,
                    obj_verts_anim,
                    color=colors[j],
                )
                # animate_rigid_mesh(f"{unq_id}6DoF{i}_{extract_obj_loc_from}", obj_faces, obj_verts, obj_params["global_orient"], obj_params["transl"], color=colors[j])

        CollectionManager.organize_collections(f"{unq_id}6DoF")

    #################
    # Visualize predicted contacts
    #################
    if smpldata_lst[0].contact is not None:
        for i, smpldata in enumerate(smpldata_lst):
            if smpldata.contact is None:
                # No contact data
                continue
            animate_predicted_contacts(
                smpldata,
                vertices=verts_lst[i],
                th_pred=0.5,
                color=(1, 0, 0, 1),
                unq_id=f"{unq_id}contacts{i}",
            )
        CollectionManager.organize_collections(f"{unq_id}contacts")

    #################
    # Visualize contact pairs
    #################
    if contact_pairs_seq is not None:
        if isinstance(contact_pairs_seq, ContactPairsSequence):
            contact_pairs_seq = [contact_pairs_seq]
        else:
            assert isinstance(contact_pairs_seq, list)

        for i, contact_pairs_seq in enumerate(contact_pairs_seq):
            contact_pairs_seq = contact_pairs_seq.to("cpu")
            offset = torch.tensor([5 + 5 * i, 0, 0])
            unq_id = f"{unq_id}ContactPairs{i}"
            animate_contact_pair_sequence(
                contact_pairs_seq,
                object_name=object_path_or_name,
                reset_blender=False,
                save_path=None,
                unq_id=unq_id,
                offset_all_but_6dof=offset,
            )

        CollectionManager.organize_collections(f"{unq_id}ContactPairs")
    #################
    # Visualize real object
    #################
    if grab_seq_path is not None:
        animate_grab_object_and_hands(
            grab_seq_path,
            save_path=None,
            tgt_fps=20,
            start_frame=start_frame,
            end_frame=start_frame + n_frames if start_frame is not None else None,
            reset_blender=False,
            translation=translation_obj,
            blender_suffix=f"{unq_id}RealGrab",
        )
        CollectionManager.organize_collections(f"{unq_id}RealGrab")

    #################
    # Visualize contact records
    #################
    if contact_record is not None:
        animate_contact_record(contact_record.to("cpu"))

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = n_frames
    scene.render.fps = 20

    # #################
    # # Organize Collections
    # #################
    # substr_map = {f"RealGrab{unq_id}": f"RealGrab{unq_id}", f"{unq_id}Contact_Pairs": f"{unq_id}Contact_Pairs"}
    # for i in range(len(smpldata_lst)):
    #     substr_map[f"sample{i}"] = f"sample{i}"
    # CollectionManager.organize_collections(f"{unq_id}sample")

    #################
    # Hide real hands
    #################
    # obj_names_to_hide = ['Mesh_Mano_rightRealGrab', 'Mesh_Mano_leftRealGrab', "Mesh_ObjectRealGrab"]
    # for obj_name in obj_names_to_hide:
    #     obj = bpy.data.objects.get(obj_name)
    #     if obj is not None:
    #         obj.hide_set(True)
    #         obj.hide_render = True

    #################
    # Hide stick figure if human mesh is available
    #################
    if AnimationSetup.MESH_ALL == anim_setup:
        for collection in [
            c for c in bpy.data.collections if "Cylinder_Human" in c.name
        ]:
            for obj in collection.objects:
                obj.hide_set(True)
                obj.hide_render = True

    #################
    # Hide mat_lst
    #################
    # if mat_lst is not None:
    #     for collection in [c for c in bpy.data.collections if "XYZ_Axes" in c.name]:
    #         collection.hide_viewport = True
    #         collection.hide_render = True

    #################
    # Add text metadata
    #################
    if text_config is not None:
        text_block = bpy.data.texts.new("Metadata")
        text_block.write(text_config)

    #################
    # Save animation
    #################
    # filename
    if save:
        if save_path is None:
            fname_parts = [
                model_id_str,
                text.replace(".", "").replace(" ", "_").replace("[SEP]", "-SEP-"),
                time.strftime("%Y%m%d_%H%M%S"),
            ]
            fname_parts = [e for e in fname_parts if e not in [None, ""]]
            fname = "__".join(fname_parts)
            save_path = os.path.realpath(os.path.join(TMP_DIR, f"{fname}.blend"))

        save_blender_file(save_path)
        blend_scp_and_run(save_path)
        return save_path

    toc = time.time()
    print(f"Hoi animation creation took {toc - tic} seconds")
