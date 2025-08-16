from collections import defaultdict
from typing import List
import numpy as np
import bpy
import torch

from hoidini.blender_utils.general_blender_utils import CollectionManager
from hoidini.datasets.grab.grab_object_records import ContactRecord

# from blender_utils.visualize_stick_figure_blender import CollectionManager
from hoidini.datasets.grab.grab_utils import get_MANO_SMPLX_vertex_ids
from hoidini.datasets.smpldata import SmplData
from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info


def add_contact_spheres(
    n_spheres: int,
    collection_name: str,
    objects_prefix: str,
    color: tuple | List[tuple] = (1, 0, 0, 1),
    radius: float = 0.008,
):
    # Add contact spheres
    assert collection_name not in bpy.data.collections
    CollectionManager.create_collection(collection_name)
    sphere_contacts = []
    for sphere_ind in range(n_spheres):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius,
            calc_uvs=False,
            enter_editmode=False,
            align="WORLD",
            scale=(1, 1, 1),
        )
        sphere_contact_name = f"{objects_prefix}_{sphere_ind}"
        sphere_object = bpy.context.object
        sphere_object.name = sphere_contact_name
        CollectionManager.add_object_to_collection(sphere_contact_name, collection_name)
        sphere_contacts.append(sphere_object)

        # color the spheres
        if color is not None:
            if isinstance(color, tuple):
                c = color
                c_name = "all_anchors"
            else:
                c = color[sphere_ind]
                c_name = f"anchor_{sphere_ind}"
            material_name = f"Mat_{collection_name}_{c_name}"
            material = bpy.data.materials.get(material_name)
            if material is None:
                # print(f"Creating material {material_name}")
                material = bpy.data.materials.new(name=material_name)
                material.diffuse_color = c if len(c) == 4 else c + (1.0,)

            sphere_object.data.materials.append(material)

    return sphere_contacts


def animate_predicted_contacts(
    smpldata: SmplData,
    vertices: np.ndarray | torch.Tensor,
    th_pred: float = 0.5,
    color: tuple = (1, 0, 0, 1),
    unq_id: str = "",
):
    anchor_inds_R2hand, _, _ = get_contact_anchors_info()
    mano_smplx_vertex_ids = get_MANO_SMPLX_vertex_ids()

    anchors_SMPLX = defaultdict(list)
    for hand in ["left_hand", "right_hand"]:
        for ind_MANO in anchor_inds_R2hand:
            ind_SMPLX = mano_smplx_vertex_ids[hand][ind_MANO]
            anchors_SMPLX[hand].append(ind_SMPLX)

    contact_one_hot = smpldata.contact
    max_contacts = contact_one_hot.shape[1]
    sphere_contacts = add_contact_spheres(
        max_contacts, f"PredictedContacts{unq_id}", f"contact_pred{unq_id}", color=color
    )
    for frame in range(len(smpldata)):
        for anchor_ind in range(max_contacts):
            if contact_one_hot[frame, anchor_ind] > th_pred:
                hand = "left_hand" if anchor_ind < (max_contacts // 2) else "right_hand"
                anchor_ind_hand = anchor_ind % (max_contacts // 2)
                anchor_ind_SMPLX = anchors_SMPLX[hand][anchor_ind_hand]
                location = vertices[frame, anchor_ind_SMPLX]
                sphere_contacts[anchor_ind].location = location.detach().cpu().numpy()
                sphere_contacts[anchor_ind].hide_viewport = False
                sphere_contacts[anchor_ind].keyframe_insert(
                    data_path="location", frame=frame
                )
                sphere_contacts[anchor_ind].keyframe_insert(
                    data_path="hide_viewport", frame=frame
                )
            else:
                sphere_contacts[anchor_ind].hide_viewport = True
                sphere_contacts[anchor_ind].keyframe_insert(
                    data_path="hide_viewport", frame=frame
                )


def animate_contact_record(
    contact_record: ContactRecord, th_force: float = 0, color: tuple = (0, 1, 0, 1)
):

    hands_verts = {
        "left": contact_record.lhand_vert_locs,
        "right": contact_record.rhand_vert_locs,
    }
    hand_contact_force = {
        "left": contact_record.lhand_contact_force,
        "right": contact_record.rhand_contact_force,
    }

    max_contacts = 0
    contact_locs_seq = []
    for frame in range(len(contact_record)):
        frame_contact_locs = []
        for hand in ["left", "right"]:
            contact_mask = hand_contact_force[hand][frame] > th_force
            frame_contact_locs.append(hands_verts[hand][frame, contact_mask])
        frame_contact_locs = np.concatenate(frame_contact_locs, axis=0)
        contact_locs_seq.append(frame_contact_locs)
        max_contacts = max(max_contacts, frame_contact_locs.shape[0])

    sphere_contacts = add_contact_spheres(
        max_contacts, "RealContacts", "contact_real", color=color
    )

    # place the spheres
    for frame in range(len(contact_record)):
        for ind in range(max_contacts):
            sphere_contact = sphere_contacts[ind]
            if ind < len(contact_locs_seq[frame]):
                loc = contact_locs_seq[frame][ind]
                sphere_contact.location = loc
                sphere_contact.hide_viewport = False
                sphere_contact.keyframe_insert(data_path="location", frame=frame)
                sphere_contact.keyframe_insert(data_path="hide_viewport", frame=frame)
            else:
                sphere_contact.hide_viewport = True
                sphere_contact.keyframe_insert(data_path="hide_viewport", frame=frame)
