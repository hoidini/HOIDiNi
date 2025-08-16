import numpy as np
from smplx.joint_names import JOINT_NAMES as SMPLX_ALL_JOINS
from skeletons.smplx_52 import BONES_52_NAMES


RIGHT_HAND_JOINT_NAMES = [
    "right_wrist",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

len(SMPLX_ALL_JOINS)

RIGHT_HAND_JOINT_INDS = np.array(
    [SMPLX_ALL_JOINS.index(jname) for jname in RIGHT_HAND_JOINT_NAMES]
)

BONE_NAMES_RIGHT_HAND = [
    (h, t) for h, t in BONES_52_NAMES if h in RIGHT_HAND_JOINT_NAMES
]
BONE_LIST_RIGHT_HAND = [
    (RIGHT_HAND_JOINT_NAMES.index(h), RIGHT_HAND_JOINT_NAMES.index(t))
    for h, t in BONE_NAMES_RIGHT_HAND
]
