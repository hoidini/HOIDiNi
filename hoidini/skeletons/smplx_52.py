from typing import List
from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES_144


BONES_52_NAMES = [
    # Core body
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("neck", "left_collar"),
    ("neck", "right_collar"),
    ("left_collar", "left_shoulder"),
    ("right_collar", "right_shoulder"),
    # Arms
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    # Fingers (left hand)
    ("left_wrist", "left_thumb1"),
    ("left_thumb1", "left_thumb2"),
    ("left_thumb2", "left_thumb3"),
    ("left_wrist", "left_index1"),
    ("left_index1", "left_index2"),
    ("left_index2", "left_index3"),
    ("left_wrist", "left_middle1"),
    ("left_middle1", "left_middle2"),
    ("left_middle2", "left_middle3"),
    ("left_wrist", "left_ring1"),
    ("left_ring1", "left_ring2"),
    ("left_ring2", "left_ring3"),
    ("left_wrist", "left_pinky1"),
    ("left_pinky1", "left_pinky2"),
    ("left_pinky2", "left_pinky3"),
    # Fingers (right hand)
    ("right_wrist", "right_thumb1"),
    ("right_thumb1", "right_thumb2"),
    ("right_thumb2", "right_thumb3"),
    ("right_wrist", "right_index1"),
    ("right_index1", "right_index2"),
    ("right_index2", "right_index3"),
    ("right_wrist", "right_middle1"),
    ("right_middle1", "right_middle2"),
    ("right_middle2", "right_middle3"),
    ("right_wrist", "right_ring1"),
    ("right_ring1", "right_ring2"),
    ("right_ring2", "right_ring3"),
    ("right_wrist", "right_pinky1"),
    ("right_pinky1", "right_pinky2"),
    ("right_pinky2", "right_pinky3"),
    # Legs
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot"),
    ("right_ankle", "right_foot"),
    # # Toes (left foot)
    # ('left_foot', 'left_big_toe'),
    # ('left_foot', 'left_small_toe'),
    # ('left_foot', 'left_heel'),
    # # Toes (right foot)
    # ('right_foot', 'right_big_toe'),
    # ('right_foot', 'right_small_toe'),
    # ('right_foot', 'right_heel'),
    # Facial bones
    # ('jaw', 'nose'),
    # ('nose', 'right_eye'),
    # ('nose', 'left_eye'),
    # ('right_eye', 'right_eye_smplhf'),
    # ('left_eye', 'left_eye_smplhf'),
    # ('right_eye', 'right_ear'),
    # ('left_eye', 'left_ear'),
]


RELEVANT_JOINT_INDS_AND_NAMES = [
    # SMPL (22)
    (0, "pelvis"),
    (1, "left_hip"),
    (2, "right_hip"),
    (3, "spine1"),
    (4, "left_knee"),
    (5, "right_knee"),
    (6, "spine2"),
    (7, "left_ankle"),
    (8, "right_ankle"),
    (9, "spine3"),
    (10, "left_foot"),
    (11, "right_foot"),
    (12, "neck"),
    (13, "left_collar"),
    (14, "right_collar"),
    (15, "head"),
    (16, "left_shoulder"),
    (17, "right_shoulder"),
    (18, "left_elbow"),
    (19, "right_elbow"),
    (20, "left_wrist"),
    (21, "right_wrist"),
    # left hand (15)
    (25, "left_index1"),
    (26, "left_index2"),
    (27, "left_index3"),
    (28, "left_middle1"),
    (29, "left_middle2"),
    (30, "left_middle3"),
    (31, "left_pinky1"),
    (32, "left_pinky2"),
    (33, "left_pinky3"),
    (34, "left_ring1"),
    (35, "left_ring2"),
    (36, "left_ring3"),
    (37, "left_thumb1"),
    (38, "left_thumb2"),
    (39, "left_thumb3"),
    # (right hand 15)
    (40, "right_index1"),
    (41, "right_index2"),
    (42, "right_index3"),
    (43, "right_middle1"),
    (44, "right_middle2"),
    (45, "right_middle3"),
    (46, "right_pinky1"),
    (47, "right_pinky2"),
    (48, "right_pinky3"),
    (49, "right_ring1"),
    (50, "right_ring2"),
    (51, "right_ring3"),
    (52, "right_thumb1"),
    (53, "right_thumb2"),
    (54, "right_thumb3"),
]

JOINT_INDS_PER_PART = {
    "lhand": range(25, 40),
    "rhand": range(40, 55),
}

assert len(RELEVANT_JOINT_INDS_AND_NAMES) == 52
SMPLX_52_144_INDS = [e[0] for e in RELEVANT_JOINT_INDS_AND_NAMES]
SMPLX_JOINT_NAMES_52 = [e[1] for e in RELEVANT_JOINT_INDS_AND_NAMES]
BONE_52_52_INDS = [
    (SMPLX_JOINT_NAMES_52.index(e[0]), SMPLX_JOINT_NAMES_52.index(e[1]))
    for e in BONES_52_NAMES
]

for j1, j2 in BONE_52_52_INDS:
    assert (SMPLX_JOINT_NAMES_52[j1], SMPLX_JOINT_NAMES_52[j2]) in BONES_52_NAMES


# inds with respect to the 144 joints
BONES_52_144_INDS = [
    (SMPLX_JOINT_NAMES_144.index(e[0]), SMPLX_JOINT_NAMES_144.index(e[1]))
    for e in BONES_52_NAMES
]
PARENT_INDS_52 = {child: parent for parent, child in BONE_52_52_INDS}


def get_joint_chain(joint, parents_dict) -> List[int]:
    chain_reversed = [joint]
    while joint in parents_dict:
        chain_reversed.append(parents_dict[joint])
        joint = parents_dict[joint]
    assert joint in [0, "pelvis"]
    chain = chain_reversed[::-1]
    return chain


WRIST_CHAINS_52 = {
    "left": get_joint_chain(SMPLX_JOINT_NAMES_52.index("left_wrist"), PARENT_INDS_52),
    "right": get_joint_chain(SMPLX_JOINT_NAMES_52.index("right_wrist"), PARENT_INDS_52),
}


def main():
    t2m_kinematic_chain = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]
    t2m_left_hand_chain = [
        [20, 22, 23, 24],
        [20, 34, 35, 36],
        [20, 25, 26, 27],
        [20, 31, 32, 33],
        [20, 28, 29, 30],
    ]
    t2m_right_hand_chain = [
        [21, 43, 44, 45],
        [21, 46, 47, 48],
        [21, 40, 41, 42],
        [21, 37, 38, 39],
        [21, 49, 50, 51],
    ]

    chains_lsts = [t2m_kinematic_chain, t2m_left_hand_chain, t2m_right_hand_chain]
    bones = []
    for chain_lst in chains_lsts:
        for chain in chain_lst:
            for i in range(len(chain) - 1):
                bones.append((chain[i], chain[i + 1]))

    print(len(bones))
    print(bones)
    set_of_joints = list(set(sum([list(b) for b in bones], [])))
    print("#joints =", len(set_of_joints))

    print(len(SMPLX_JOINT_NAMES_144))

    print("relevant_joints", [j for j in SMPLX_JOINT_NAMES_144 if j in set_of_joints])

    BONE_NAMES = [
        (SMPLX_JOINT_NAMES_144[i], SMPLX_JOINT_NAMES_144[j]) for i, j in bones
    ]
    print(BONE_NAMES)
