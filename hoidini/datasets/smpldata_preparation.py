""" "
TODO: Move all code fragments that create SmplData to here
"""

import numpy as np
import torch
from tqdm import tqdm
import os
from typing import Dict, List

from hoidini.datasets.grab.grab_utils import (
    get_MANO_SMPLX_vertex_ids,
    grab_seq_path_to_unique_name,
    parse_npz,
    reduce_seq_data,
    grab_seq_data_to_object_name,
)
import hoidini.smplx as smplx
from hoidini.datasets.smpldata import SmplData
from hoidini.general_utils import (
    PROJECT_DIR,
    SRC_DIR,
    numpify_torch_dict,
    torchify_numpy_dict,
)
from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info
from hoidini.skeletons.smplx_52 import SMPLX_52_144_INDS, SMPLX_JOINT_NAMES_52
from hoidini.datasets.smpldata import SMPL_MODELS_PATH

intent_list = [
    "offhand",
    "pass",
    "lift",
    "drink",
    "brush",
    "eat",
    "peel",
    "takepicture",
    "see",
    "wear",
    "play",
    "clean",
    "browse",
    "inspect",
    "pour",
    "use",
    "switchON",
    "cook",
    "toast",
    "staple",
    "squeeze",
    "set",
    "open",
    "chop",
    "screw",
    "call",
    "shake",
    "fly",
    "stamp",
]


MAPPING_INTENT = {intent: idx for idx, intent in enumerate(intent_list)}


def get_grab_extended_smpldata_lst(
    grab_seq_paths: str, fk_device: str = "cuda", contact_threshold: float = 0.0
) -> List[Dict]:
    """
    fk_device: device for forward kinematics, anyway the output is moved to cpu
    """
    anchor_inds_R2hand, _, closest_anchor_per_vertex_R2hands = (
        get_contact_anchors_info()
    )
    mano_smplx_vertex_ids = get_MANO_SMPLX_vertex_ids()

    results = []
    for grab_seq_path in tqdm(grab_seq_paths, desc="Grab to SmplData"):
        result = get_extended_smpldata(
            grab_seq_path,
            fk_device,
            contact_threshold,
            anchor_inds_R2hand,
            closest_anchor_per_vertex_R2hands,
            mano_smplx_vertex_ids,
        )
        results.append(result)

    return results


def get_extended_smpldata(
    grab_seq_path,
    fk_device="cpu",
    contact_threshold=0.1,
    anchor_inds_R2hand=None,
    closest_anchor_per_vertex_R2hands=None,
    mano_smplx_vertex_ids=None,
):
    if anchor_inds_R2hand is None:
        anchor_inds_R2hand, _, closest_anchor_per_vertex_R2hands = (
            get_contact_anchors_info()
        )
    if closest_anchor_per_vertex_R2hands is None:
        _, _, closest_anchor_per_vertex_R2hands = get_contact_anchors_info()
    if mano_smplx_vertex_ids is None:
        mano_smplx_vertex_ids = get_MANO_SMPLX_vertex_ids()

    seq_data = parse_npz(grab_seq_path)
    seq_data = reduce_seq_data(seq_data, tgt_fps=20)

    sbj_model: smplx.SMPLX = smplx.create(
        model_path=SMPL_MODELS_PATH,
        model_type="smplx",
        gender="neutral",
        create_jaw_pose=True,
        batch_size=seq_data["n_frames"],
        use_pca=False,
        flat_hand_mean=True,
    ).to(fk_device)

    smplx_input = {
        "transl": seq_data["body"]["params"]["transl"],
        "global_orient": seq_data["body"]["params"]["global_orient"],
        "body_pose": seq_data["body"]["params"]["body_pose"],
        "left_hand_pose": seq_data["lhand"]["params"]["fullpose"],
        "right_hand_pose": seq_data["rhand"]["params"]["fullpose"],
    }
    smplx_input = torchify_numpy_dict(smplx_input, device=fk_device, dtype=torch.float)
    with torch.no_grad():
        smplx_output = sbj_model(**smplx_input)

    # #########################
    # Build SmplData
    # #########################

    #  1) Human motion components
    joints = smplx_output.joints[:, SMPLX_52_144_INDS, :]
    poses = torch.concat(
        [
            smplx_input["global_orient"],
            smplx_input["body_pose"],
            smplx_input["left_hand_pose"],
            smplx_input["right_hand_pose"],
        ],
        dim=1,
    )

    # 2) object motion components
    obj_global_orient = torch.from_numpy(
        seq_data["object"]["params"]["global_orient"]
    ).to(device=fk_device, dtype=torch.float)
    obj_transl = torch.from_numpy(seq_data["object"]["params"]["transl"]).to(
        device=fk_device, dtype=torch.float
    )

    # 3) Contact related
    contact_body = torch.from_numpy(seq_data["contact"]["body"]).to(
        device=fk_device, dtype=torch.float
    )

    hand_contact_reduced_dict = {}
    for hand in ["left", "right"]:
        verts_map_fullbody2hand = torch.from_numpy(
            mano_smplx_vertex_ids[f"{hand}_hand"]
        ).to(device=fk_device, dtype=torch.long)
        hand_contact = contact_body[:, verts_map_fullbody2hand] > contact_threshold
        hand_contact_reduced = torch.zeros(
            (seq_data["n_frames"], len(anchor_inds_R2hand))
        )
        for anchor_ind_rel2anchors, anchor_ind_R2hand in enumerate(anchor_inds_R2hand):
            mask_anchor_R2hand = closest_anchor_per_vertex_R2hands == anchor_ind_R2hand
            hand_contact_reduced[:, anchor_ind_rel2anchors] = hand_contact[
                :, mask_anchor_R2hand
            ].sum(dim=1)
            # hand_contact_reduced[:, anchor_ind_rel2anchors] = hand_contact_reduced[:, anchor_ind_rel2anchors] > 0  # to bool
        hand_contact_reduced_dict[hand] = hand_contact_reduced
        assert (
            hand_contact_reduced.sum() == hand_contact.sum()
        ), "reduced contact does not match original contact"
        hand_contact_reduced_dict[hand] = (
            hand_contact_reduced > contact_threshold
        )  # to bool

    lhand_contact_reduced = hand_contact_reduced_dict["left"]
    rhand_contact_reduced = hand_contact_reduced_dict["right"]
    contact = torch.cat([lhand_contact_reduced, rhand_contact_reduced], dim=1)
    assert contact.sum() > 0

    smpldata = SmplData(
        poses=poses,
        trans=smplx_input["transl"],
        joints=joints,
        poses_obj=obj_global_orient,
        trans_obj=obj_transl,
        contact=contact,
        global_lhand_rotmat=smplx_output.global_joints_transforms[
            :, SMPLX_JOINT_NAMES_52.index("left_wrist"), :3, :3
        ],
        global_rhand_rotmat=smplx_output.global_joints_transforms[
            :, SMPLX_JOINT_NAMES_52.index("right_wrist"), :3, :3
        ],
    )

    smpldata = smpldata.to("cpu")
    intent_vec = one_hot_vectors(seq_data["motion_intent"], intent_list, MAPPING_INTENT)
    object_name = grab_seq_data_to_object_name(seq_data)
    result = {
        "smpldata": smpldata,
        "object_name": object_name,
        "intent_vec": intent_vec,
        "grab_seq_path": grab_seq_path,
    }
    return result


def one_hot_vectors(word, word_list, mapping):
    arr = list(np.zeros(len(word_list), dtype=int))
    arr[mapping[word]] = 1
    return np.array(arr)


def main():
    grab_seq_paths = [
        os.path.join(PROJECT_DIR, "DATASETS/Data_GRAB/s3/airplane_lift.npz"),
    ]
    ext_smpldata_lst = get_grab_extended_smpldata_lst(grab_seq_paths, fk_device="cpu")
    save_dir = os.path.join(SRC_DIR, "resources")
    for ext_smpldata in ext_smpldata_lst:
        seq_path = ext_smpldata["grab_seq_path"]
        smpldata = ext_smpldata["smpldata"]
        object_name = ext_smpldata["object_name"]
        intent_vec = ext_smpldata["intent_vec"]
        seq_id = grab_seq_path_to_unique_name(seq_path)

        new_path = os.path.join(save_dir, f"smpldata_w_hoi_{seq_id}.npz")
        smpldata_dict = smpldata.to_dict()
        smpldata_dict = numpify_torch_dict(smpldata_dict)
        np.savez(new_path, **smpldata_dict)


if __name__ == "__main__":
    main()
