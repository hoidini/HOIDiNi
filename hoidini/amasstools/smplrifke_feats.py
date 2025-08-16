import torch
import einops
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

# from smplx import SMPLX
# from smplx.lbs import batch_rigid_transform


from hoidini.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)
from hoidini.datasets.smpldata import SmplData
from hoidini.skeletons.smplx_52 import SMPLX_JOINT_NAMES_52, WRIST_CHAINS_52


FEATURES_STRINGS = {
    "human": "root_height|root_lin_vel|root_rot_vel|poses|joints",
    "hoi": "root_height|root_lin_vel|root_rot_vel|poses|joints|poses_obj_r2b|trans_obj_r2b|contact",
    "hoi_body_hands": "root_height|root_lin_vel|root_rot_vel|poses|joints|poses_obj_r2b|trans_obj_r2b|poses_obj_r2lh|trans_obj_r2lh|poses_obj_r2rh|trans_obj_r2rh|contact",
    "hoi_body_hands_world": "root_height|root_lin_vel|root_rot_vel|poses|joints|poses_obj_r2b|trans_obj_r2b|poses_obj_r2lh|trans_obj_r2lh|poses_obj_r2rh|trans_obj_r2rh|poses_obj_r2world|trans_obj_r2world|contact",
    "human_w_contact": "root_height|root_lin_vel|root_rot_vel|poses|joints|contact",
    "human_w_any_contact": "root_height|root_lin_vel|root_rot_vel|poses|joints|any_contact",
    "cphoi": "root_height|root_lin_vel|root_rot_vel|poses|joints|local_object_points|contact|object_poses_cont6d|object_trans_global|object_velocity_global",
    "hoi_global": "joints_global|object_poses_cont6d|object_trans_global|object_velocity_global",  # TODO: implement
}


# Feature registry with names, sizes, and metadata
class FeatureType(Enum):
    """
    R2<X> - Relative to <X>
    """

    # Basic human features (always available)
    ROOT_HEIGHT = "root_height"  # Root height above ground
    ROOT_LIN_VEL = "root_lin_vel"  # 2D trajectory velocity
    ROOT_ROT_VEL = "root_rot_vel"  # Rotation velocity around vertical axis
    POSES = "poses"  # Body pose features (6D representation)
    JOINTS = "joints"  # Joint positions relative to root

    # Object features
    POSES_OBJ_R2B = (
        "poses_obj_r2b"  # Object orientation (6D representation) relative to body
    )
    TRANS_OBJ_R2B = "trans_obj_r2b"  # Object translation relative to body

    POSES_OBJ_R2LH = (
        "poses_obj_r2lh"  # Object orientation (6D representation) relative to left hand
    )
    TRANS_OBJ_R2LH = "trans_obj_r2lh"  # Object translation relative to left hand

    POSES_OBJ_R2RH = "poses_obj_r2rh"  # Object orientation (6D representation) relative to right hand
    TRANS_OBJ_R2RH = "trans_obj_r2rh"  # Object translation relative to right hand

    POSES_OBJ_R2WORLD = "poses_obj_r2world"  # Object orientation (6D representation) similar to the root
    TRANS_OBJ_R2WORLD = "trans_obj_r2world"  # Object translation similar to the root

    # Contact
    CONTACT = "contact"  # Binary flags indicating contact state for anchor vertices

    ANY_CONTACT = "any_contact"  # Binary flag indicating contact state

    LOCAL_OBJECT_POINTS = "local_object_points"  # Local object points
    OBJECT_POSES_CONT6D = "object_poses_cont6d"  # Object poses in 6D representation
    OBJECT_TRANS_GLOBAL = (
        "object_trans_global"  # Object translation in global coordinates
    )
    OBJECT_VELOCITY_GLOBAL = (
        "object_velocity_global"  # Object velocity in global coordinates
    )

    @staticmethod
    def from_string(feature_str: str) -> "FeatureType":
        """Convert string to FeatureType enum."""
        for feature in FeatureType:
            if feature.value == feature_str:
                return feature
        raise ValueError(f"Unknown feature: {feature_str}")

    @staticmethod
    def parse_feature_string(
        feature_string: Optional[str] = None,
    ) -> List["FeatureType"]:
        """
        Parse a string of '|' separated feature names into a list of FeatureType.
        If the feature string is in FEATURES_STRINGS, it will be used as is.
        """
        if feature_string is None:
            feature_string = "human"
        if feature_string in FEATURES_STRINGS:
            feature_string = FEATURES_STRINGS[feature_string]

        features = []
        for feature_str in feature_string.split("|"):
            feature_str = feature_str.strip()
            if feature_str:
                features.append(FeatureType.from_string(feature_str))
        return features


def get_global_hand_rotation_mat(poses: Tensor, hand: str) -> Tensor:
    """
    poses: (seq_len, n_joints * 3)
    hand: "left" or "right"
    """
    seq_len, n_joints_x3 = poses.shape
    n_joints = n_joints_x3 // 3
    pose_matrices = axis_angle_to_matrix(poses.reshape(-1, 3)).reshape(
        seq_len, n_joints, 3, 3
    )
    global_rotation = pose_matrices[:, 0]  # (seq_len, 3, 3)
    chain = WRIST_CHAINS_52[hand]
    for joint_ind in chain[1:]:
        global_rotation = torch.matmul(global_rotation, pose_matrices[:, joint_ind])
    # root @ chain[1] @ chain[2] @ ... @ chain[-1]
    return global_rotation


# Feature metadata and sizing
FEATURE_METADATA = {
    FeatureType.ROOT_HEIGHT: {
        "size_func": lambda n_pose_joints, n_contact_verts: 1,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.ROOT_LIN_VEL: {
        "size_func": lambda n_pose_joints, n_contact_verts: 2,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.ROOT_ROT_VEL: {
        "size_func": lambda n_pose_joints, n_contact_verts: 1,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.POSES: {
        "size_func": lambda n_pose_joints, n_contact_verts: n_pose_joints * 6,
        "group_reshape": lambda tensor: einops.rearrange(tensor, "k l t -> k (l t)"),
        "ungroup_reshape": lambda tensor: einops.rearrange(
            tensor, "k (l t) -> k l t", t=6
        ),
    },
    FeatureType.JOINTS: {
        "size_func": lambda n_pose_joints, n_contact_verts: (n_pose_joints - 1) * 3,
        "group_reshape": lambda tensor: einops.rearrange(tensor, "k l t -> k (l t)"),
        "ungroup_reshape": lambda tensor: einops.rearrange(
            tensor, "k (l t) -> k l t", t=3
        ),
    },
    FeatureType.POSES_OBJ_R2B: {
        "size_func": lambda n_pose_joints, n_contact_verts: 6,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.TRANS_OBJ_R2B: {
        "size_func": lambda n_pose_joints, n_contact_verts: 3,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.CONTACT: {
        "size_func": lambda n_pose_joints, n_contact_verts: n_contact_verts,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.POSES_OBJ_R2RH: {
        "size_func": lambda n_pose_joints, n_contact_verts: 6,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.TRANS_OBJ_R2RH: {
        "size_func": lambda n_pose_joints, n_contact_verts: 3,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.POSES_OBJ_R2LH: {
        "size_func": lambda n_pose_joints, n_contact_verts: 6,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.TRANS_OBJ_R2LH: {
        "size_func": lambda n_pose_joints, n_contact_verts: 3,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.POSES_OBJ_R2WORLD: {
        "size_func": lambda n_pose_joints, n_contact_verts: 6,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.TRANS_OBJ_R2WORLD: {
        "size_func": lambda n_pose_joints, n_contact_verts: 3,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.LOCAL_OBJECT_POINTS: {
        "size_func": lambda n_pose_joints, n_contact_verts: 3 * n_contact_verts,
        "group_reshape": lambda tensor: einops.rearrange(tensor, "k l t -> k (l t)"),
        "ungroup_reshape": lambda tensor: einops.rearrange(
            tensor, "k (l t) -> k l t", t=3
        ),
    },
    FeatureType.OBJECT_POSES_CONT6D: {
        "size_func": lambda n_pose_joints, n_contact_verts: 6,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.OBJECT_TRANS_GLOBAL: {
        "size_func": lambda n_pose_joints, n_contact_verts: 3,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
    FeatureType.OBJECT_VELOCITY_GLOBAL: {
        "size_func": lambda n_pose_joints, n_contact_verts: 3,
        "group_reshape": None,
        "ungroup_reshape": None,
    },
}


class SMPLFeatureProcessor:
    """
    Processor for converting between SMPL data and feature representations.

    This class handles the transformation between SMPL data dictionaries and
    compact feature representations that can be used for ML models.
    """

    def __init__(
        self,
        n_pose_joints: int = 52,
        n_contact_verts: Optional[int] = None,
        features_string: Optional[str] = None,
        th_clamp_rel2hand: Optional[float] = 3.0,
    ):
        """
        Initialize the SMPL feature processor.

        Args:
            n_pose_joints: Number of pose joints in the SMPL model
            n_contact_verts: Number of contact vertices when using object features
            features_string: String of '|' separated feature names to include by default or a key from FEATURES_STRINGS
            th_clamp_rel2hand: Threshold to clamp the translation values of the object relative to the hand
        """
        self.n_pose_joints = n_pose_joints
        self.n_contact_verts = n_contact_verts
        self.features_string = features_string
        self.th_clamp_rel2hand = th_clamp_rel2hand

        self.features = FeatureType.parse_feature_string(self.features_string)
        self.sizes = self.get_sizes()

    def get_sizes(self) -> List[int]:
        sizes = []
        for feature in self.features:
            size = FEATURE_METADATA[feature]["size_func"](
                self.n_pose_joints, self.n_contact_verts
            )
            sizes.append(size)
        return sizes

    def get_total_feature_size(self) -> int:
        return sum(self.sizes)

    def is_fk_required(self) -> bool:
        """
        Check if FK is required after decoding the features
        """
        if FeatureType.POSES_OBJ_R2B in self.features:
            return True
        if FeatureType.TRANS_OBJ_R2B in self.features:
            return True
        if FeatureType.POSES_OBJ_R2LH in self.features:
            return True
        if FeatureType.TRANS_OBJ_R2LH in self.features:
            return True
        if FeatureType.POSES_OBJ_R2RH in self.features:
            return True
        if FeatureType.TRANS_OBJ_R2RH in self.features:
            return True
        return False

    def get_features_string_vector(self) -> np.ndarray:
        results = []
        for feature, size in zip(self.features, self.sizes):
            results += [feature.value] * size
        return np.array(results)

    def get_feature_mask(self, feature_name: str) -> Tensor:
        """
        Get a mask for the feature name
        """
        return torch.from_numpy(self.get_features_string_vector() == feature_name)

    def get_object_related_feature_mask(self) -> Tensor:
        object_related_feature_names = []
        for feature in self.features:
            if feature.value not in FEATURES_STRINGS["human"]:
                object_related_feature_names.append(feature.value)
        feature_name_vec = self.get_features_string_vector()
        mask = torch.zeros(len(feature_name_vec), dtype=torch.bool)
        for feature_name in object_related_feature_names:
            mask = mask | torch.from_numpy(feature_name_vec == feature_name)
        return mask

    def encode(
        self, smpldata: SmplData, return_tfms: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Convert SMPL data to feature representation.

        Args:
            smpldata: SmplData object

        Returns:
            Tuple of:
                - Packed features tensor
                - Root transformation matrix (4x4)
        """
        ############################################################
        # will be filled along the way
        features_dict = {}
        ############################################################
        smpldata = smpldata.to_dict()

        # Process data
        poses = smpldata["poses"].clone()
        trans = smpldata["trans"].clone()
        joints = smpldata["joints"].clone()

        # Check shapes
        assert poses.shape[-1] == self.n_pose_joints * 3
        assert len(poses.shape) == 2
        assert len(trans.shape) == 2
        assert len(joints.shape) == 3, "Joints should have shape (seq_len, n_joints, 3)"
        assert joints.shape[-1] == 3, "Last dimension of joints should be 3 (XYZ)"
        if smpldata["poses_obj"] is not None:
            assert len(smpldata["poses_obj"].shape) == 2
            assert len(smpldata["trans_obj"].shape) == 2
        if smpldata["contact"] is not None:
            assert len(smpldata["contact"].shape) == 2

        # JOINTS PROCESS

        # First remove the ground
        ground = joints[:, :, 2].min()
        joints[:, :, 2] -= ground

        root_grav_axis = joints[:, 0, 2].clone()
        # Add to features dict with the correct enum value
        features_dict[FeatureType.ROOT_HEIGHT.value] = root_grav_axis

        # Make sure values are consistent
        _val = abs(
            (trans[:, 2] - trans[0, 2]) - (root_grav_axis - root_grav_axis[0])
        ).mean()
        assert _val < 1e-6, _val

        # Trajectory => Translation without gravity axis (Z)
        trajectory = joints[:, 0, :2].clone()

        # Make sure values are consistent
        _val = torch.abs(
            (trajectory - trajectory[0]) - (trans[:, :2] - trans[0, :2])
        ).mean()
        assert _val < 1e-6, _val

        # Joints in the pelvis coordinate system
        joints[:, :, [0, 1]] -= trajectory[..., None, :]
        # Also doing it for the Z coordinate
        joints[:, :, 2] -= joints[:, [0], 2]

        # check that the pelvis is all zero
        assert (joints[:, 0] == 0).all()

        # Delete the pelvis from the local representation
        # it is already encoded in root_grav_axis and trajectory
        joints = joints[:, 1:]

        vel_trajectory = torch.diff(trajectory, dim=0)
        # repeat the last acceleration
        # for the last (not seen) velocity
        last_acceleration = vel_trajectory[-1] - vel_trajectory[-2]
        future_velocity = vel_trajectory[-1] + last_acceleration
        vel_trajectory = torch.cat((vel_trajectory, future_velocity[None]), dim=0)

        # SMPL PROCESS

        # unflatten
        poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
        poses_mat = axis_angle_to_matrix(poses)

        global_orient = poses_mat[:, 0]
        # Decompose the rotation into 3 euler angles rotations
        # To extract and remove the Z rotation for each frames
        global_euler = matrix_to_euler_angles(global_orient, "ZYX")
        rotZ_angle, rotY_angle, rotX_angle = torch.unbind(global_euler, -1)

        # Construct the rotations matrices
        rotZ = axis_angle_rotation("Z", rotZ_angle)
        rotY = axis_angle_rotation("Y", rotY_angle)
        rotX = axis_angle_rotation("X", rotX_angle)

        # check the reconstruction
        global_orient_recons = rotZ @ rotY @ rotX
        # sanity check
        assert torch.abs(global_orient - global_orient_recons).mean() < 1e-6

        # Construct root transformation matrix
        tfms = torch.eye(4, device=rotZ.device).unsqueeze(0).repeat(rotZ.shape[0], 1, 1)
        tfms[:, :3, :3] = rotZ  # Set rotation
        tfms[:, :3, 3] = torch.cat(
            [trajectory, ground.expand(trajectory.shape[0], 1)], dim=-1
        )
        # tfms[:, :3, 3] = torch.cat([trajectory, joints[:, 0, [2]]], dim=-1)
        # tfms[:, :3, 3] = joints[:, 0]

        # construct the local global pose
        # the one without the final Z rotation
        global_orient_local = rotY @ rotX

        # True difference of angles
        # robust way of computing torch.diff with angles
        vel_rotZ = rotZ[1:] @ rotZ.transpose(1, 2)[:-1]
        # repeat the last acceleration (same as the trajectory but in the 3D
        # rotation space)
        last_acc_rotZ = vel_rotZ[-1] @ vel_rotZ.transpose(1, 2)[-2]
        future_vel_rotZ = vel_rotZ[-1] @ last_acc_rotZ
        vel_rotZ = torch.cat((vel_rotZ, future_vel_rotZ[None]), dim=-3)
        vel_angles = matrix_to_axis_angle(vel_rotZ)[:, 2]
        # Add to features dict with the correct enum value
        features_dict[FeatureType.ROOT_ROT_VEL.value] = vel_angles

        # Rotate the vel_trajectory (rotation inverse in the indexes)
        vel_trajectory_local = torch.einsum(
            "tkj,tk->tj", rotZ[:, :2, :2], vel_trajectory
        )
        # Add to features dict with the correct enum value
        features_dict[FeatureType.ROOT_LIN_VEL.value] = vel_trajectory_local

        # Rotate the local_joints
        joints_local = torch.einsum(
            "tkj,tlk->tlj", rotZ[:, :2, :2], joints[:, :, [0, 1]]
        )
        joints_local = torch.stack(
            (joints_local[..., 0], joints_local[..., 1], joints[..., 2]), axis=-1
        )
        # Add to features dict with the correct enum value
        features_dict[FeatureType.JOINTS.value] = joints_local

        # Replace the global orient with the one without rotation
        poses_mat_local = torch.cat(
            (global_orient_local[:, None], poses_mat[:, 1:]), dim=1
        )
        poses_local = matrix_to_rotation_6d(poses_mat_local)
        # Add to features dict with the correct enum value
        features_dict[FeatureType.POSES.value] = poses_local

        # Process object if present in input data
        if (
            FeatureType.POSES_OBJ_R2B in self.features
            or FeatureType.TRANS_OBJ_R2B in self.features
        ):
            # Convert object pose to rotation matrix
            pose_obj = smpldata["poses_obj"].clone()
            pose_obj_mat = axis_angle_to_matrix(pose_obj.view(-1, 1, 3))[:, 0]

            # Make object rotation relative to root
            pose_obj_mat = rotZ.transpose(1, 2) @ pose_obj_mat

            # Convert to 6D representation
            pose_obj = matrix_to_rotation_6d(pose_obj_mat)

            # Make object translation relative to root
            trans_obj_r2b = smpldata["trans_obj"].clone()
            trans_obj_r2b[:, :2] -= trajectory  # XY relative to trajectory
            trans_obj_r2b[:, 2] -= root_grav_axis  # Z relative to root height

            # UPDATED: Use the same approach as used for joints
            # Rotate translation to be in root-aligned frame - matching the joints pattern
            trans_obj_xy = torch.einsum(
                "tkj,tk->tj", rotZ[:, :2, :2], trans_obj_r2b[:, :2]
            )
            trans_obj_r2b = torch.cat([trans_obj_xy, trans_obj_r2b[:, 2:3]], dim=-1)

            # Add object data to features dict with the correct enum values
            features_dict[FeatureType.POSES_OBJ_R2B.value] = pose_obj
            features_dict[FeatureType.TRANS_OBJ_R2B.value] = trans_obj_r2b

        if (
            FeatureType.POSES_OBJ_R2WORLD in self.features
            or FeatureType.TRANS_OBJ_R2WORLD in self.features
        ):
            # Convert object pose to rotation matrix
            pose_obj = smpldata["poses_obj"].clone()
            pose_obj_mat = axis_angle_to_matrix(pose_obj.view(-1, 1, 3))[:, 0]

            # Make object rotation relative to root

        # Process object features relative to hands
        for hand in ["left", "right"]:
            pose_feature = (
                FeatureType.POSES_OBJ_R2LH
                if hand == "left"
                else FeatureType.POSES_OBJ_R2RH
            )
            trans_feature = (
                FeatureType.TRANS_OBJ_R2LH
                if hand == "left"
                else FeatureType.TRANS_OBJ_R2RH
            )
            global_hand_rotmat = (
                smpldata["global_lhand_rotmat"]
                if hand == "left"
                else smpldata["global_rhand_rotmat"]
            )  # (seq_len, 3, 3)
            if pose_feature in self.features or trans_feature in self.features:
                pose_obj = smpldata["poses_obj"].clone()
                global_obj_location = smpldata["trans_obj"].clone()

                global_obj_rotation_mat = axis_angle_to_matrix(pose_obj.view(-1, 1, 3))[
                    :, 0
                ]  # (seq_len, 3, 3)
                pose_obj_mat_r2hand = (
                    global_hand_rotmat.transpose(1, 2) @ global_obj_rotation_mat
                )
                pose_obj_r2hand = matrix_to_rotation_6d(pose_obj_mat_r2hand)
                features_dict[pose_feature.value] = pose_obj_r2hand

                global_hand_location = smpldata["joints"].clone()[
                    :, SMPLX_JOINT_NAMES_52.index(hand + "_wrist")
                ]  # (seq_len, 3)
                trans_obj_r2hand = torch.einsum(
                    "bij,bj->bi",
                    global_hand_rotmat.transpose(1, 2),
                    global_obj_location - global_hand_location,
                )
                if self.th_clamp_rel2hand is not None:
                    trans_obj_r2hand = torch.clamp(
                        trans_obj_r2hand,
                        min=-self.th_clamp_rel2hand,
                        max=self.th_clamp_rel2hand,
                    )
                features_dict[trans_feature.value] = trans_obj_r2hand

        # Add contact data if available
        if FeatureType.LOCAL_OBJECT_POINTS in self.features:
            features_dict[FeatureType.LOCAL_OBJECT_POINTS.value] = smpldata[
                "local_object_points"
            ].clone()

        if FeatureType.CONTACT in self.features:
            features_dict[FeatureType.CONTACT.value] = smpldata["contact"].clone()

        if FeatureType.ANY_CONTACT in self.features:
            features_dict[FeatureType.ANY_CONTACT.value] = (
                smpldata["contact"].clone().sum(dim=1) > 0
            )

        if FeatureType.OBJECT_POSES_CONT6D in self.features:

            features_dict[FeatureType.OBJECT_POSES_CONT6D.value] = axis_angle_to_cont6d(
                smpldata["poses_obj"]
            )

        if FeatureType.OBJECT_TRANS_GLOBAL in self.features:
            features_dict[FeatureType.OBJECT_TRANS_GLOBAL.value] = smpldata[
                "trans_obj"
            ].clone()

        if FeatureType.OBJECT_VELOCITY_GLOBAL in self.features:
            locs = smpldata["trans_obj"].clone()  # (seq_len, 3)
            vel = locs[1:] - locs[:-1]
            vel = torch.cat([vel, vel[-1:]], dim=0)
            features_dict[FeatureType.OBJECT_VELOCITY_GLOBAL.value] = vel

        # Stack selected features
        features = self.group(features_dict)

        if return_tfms:
            return features, tfms
        else:
            return features

    def decode(self, features: Tensor, tfm: Optional[Tensor] = None) -> SmplData:
        """
        Convert feature representation back to SMPL data.

        Args:
            features: Packed features tensor

        Returns:
            Dictionary containing SMPL data
        """

        # Ungroup features
        feature_dict = self.ungroup(features)

        # for k, v in feature_dict.items():
        #     print(k, v.shape)

        # Initialize result dictionary
        smpldata = {}

        # Get the features we need using the enum values as keys
        root_grav_axis = feature_dict[FeatureType.ROOT_HEIGHT.value]
        vel_trajectory_local = feature_dict[FeatureType.ROOT_LIN_VEL.value]
        vel_angles = feature_dict[FeatureType.ROOT_ROT_VEL.value]
        poses_local = feature_dict[FeatureType.POSES.value]

        # Get joints (which will always be available)
        joints_local = feature_dict[FeatureType.JOINTS.value]

        poses_mat_local = rotation_6d_to_matrix(poses_local)
        global_orient_local = poses_mat_local[:, 0]

        # Remove the dummy last angle and integrate the angles
        angles = torch.cumsum(vel_angles[:-1], dim=0)
        # The first angle is zero (canonicalization)
        angles = torch.cat((0 * angles[[0]], angles), dim=0)

        # Construct the rotation matrix
        rotZ = axis_angle_rotation("Z", angles)

        # Rotate the trajectory (normal rotation in the indexes)
        vel_trajectory = torch.einsum(
            "bjk,bk->bj", rotZ[:, :2, :2], vel_trajectory_local
        )

        # Process joints (which will always be available)
        joints = torch.einsum(
            "bjk,blk->blj", rotZ[:, :2, :2], joints_local[..., [0, 1]]
        )
        joints = torch.stack(
            (joints[..., 0], joints[..., 1], joints_local[..., 2]), axis=-1
        )

        # Remove the dummy last velocity and integrate the trajectory
        trajectory = torch.cumsum(vel_trajectory[..., :-1, :], dim=-2)
        # The first position is zero
        trajectory = torch.cat((0 * trajectory[..., [0], :], trajectory), dim=-2)

        # Add the pelvis (which is still zero)
        joints = torch.cat((0 * joints[:, [0]], joints), axis=1)

        # Adding back the Z component
        joints[:, :, 2] += root_grav_axis[:, None]
        # Adding back the trajectory
        joints[:, :, [0, 1]] += trajectory[:, None]

        # Add to result
        smpldata["joints"] = joints

        # Get back the translation
        trans = torch.cat([trajectory, root_grav_axis[..., None]], dim=1)
        smpldata["trans"] = trans

        # Remove the predicted Z rotation inside global_orient_local
        # It is trained to be zero, but the network could produce non zeros
        # outputs
        global_euler_local = matrix_to_euler_angles(global_orient_local, "ZYX")
        _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
        rotY = axis_angle_rotation("Y", rotY_angle)
        rotX = axis_angle_rotation("X", rotX_angle)

        # Replace it with the one computed with velocities
        global_orient = rotZ @ rotY @ rotX
        poses_mat = torch.cat(
            (global_orient[..., None, :, :], poses_mat_local[..., 1:, :, :]), dim=-3
        )

        poses = matrix_to_axis_angle(poses_mat)
        # flatten back
        poses = einops.rearrange(poses, "k l t -> k (l t)")
        smpldata["poses"] = poses

        # Process object features if included in the feature dictionary

        if (
            FeatureType.POSES_OBJ_R2B in self.features
            or FeatureType.TRANS_OBJ_R2B in self.features
        ):
            pose_obj = feature_dict[FeatureType.POSES_OBJ_R2B.value]
            trans_obj = feature_dict[FeatureType.TRANS_OBJ_R2B.value]

            # Convert object pose from 6D back to rotation matrix
            pose_obj_mat = rotation_6d_to_matrix(pose_obj)

            # Transform from local back to global coordinates
            pose_obj_mat = rotZ @ pose_obj_mat

            # Convert to axis-angle
            pose_obj = matrix_to_axis_angle(pose_obj_mat)

            # UPDATED: Use the same transformation as for joints
            # Transform translation back to global coordinates
            trans_obj_xy = torch.einsum("bjk,bk->bj", rotZ[:, :2, :2], trans_obj[:, :2])
            trans_obj = torch.cat([trans_obj_xy, trans_obj[:, 2:3]], dim=-1)

            # Add back root trajectory and height
            trans_obj[:, :2] += trajectory
            trans_obj[:, 2] += root_grav_axis

            # Add object data to output
            smpldata["poses_obj"] = pose_obj
            smpldata["trans_obj"] = trans_obj

        for hand in ["left", "right"]:
            pose_feature = (
                FeatureType.POSES_OBJ_R2LH
                if hand == "left"
                else FeatureType.POSES_OBJ_R2RH
            )
            trans_feature = (
                FeatureType.TRANS_OBJ_R2LH
                if hand == "left"
                else FeatureType.TRANS_OBJ_R2RH
            )
            if pose_feature in self.features or trans_feature in self.features:
                pose_obj_6d_r2hand = feature_dict[pose_feature.value]
                pose_obj_mat_r2hand = rotation_6d_to_matrix(pose_obj_6d_r2hand)
                smpldata[f"_rotmat_obj_rel2_{hand[:1]}hand"] = pose_obj_mat_r2hand
                smpldata[f"_trans_obj_rel2_{hand[:1]}hand"] = feature_dict[
                    trans_feature.value
                ]

        if FeatureType.CONTACT in self.features:
            smpldata["contact"] = feature_dict[FeatureType.CONTACT.value]

        if FeatureType.OBJECT_TRANS_GLOBAL in self.features:
            decode_global_obj_pose_strategy = "vel_corr"
            ema_alpha = 0.2
            vel_corr_beta = 0.1
            vel = feature_dict["object_velocity_global"]
            abs_pred = feature_dict[FeatureType.OBJECT_TRANS_GLOBAL.value]
            obj_trans = decode_global_trans(
                decode_global_obj_pose_strategy, ema_alpha, vel_corr_beta, vel, abs_pred
            )
            smpldata["trans_obj"] = obj_trans

        if FeatureType.OBJECT_POSES_CONT6D in self.features:
            smpldata["poses_obj"] = cont6d_to_axis_angle(
                feature_dict[FeatureType.OBJECT_POSES_CONT6D.value]
            )

        if FeatureType.LOCAL_OBJECT_POINTS in self.features:
            smpldata["local_object_points"] = feature_dict[
                FeatureType.LOCAL_OBJECT_POINTS.value
            ]

        if tfm is not None:
            device = smpldata["joints"].device
            tfm = tfm.to(device)  # (4, 4)
            assert tfm.shape == (4, 4), f"Expected tfm shape (4, 4), got {tfm.shape}"

            # Extract rotation and translation for the initial frame
            Rg = tfm[:3, :3]  # (3, 3)
            tg = tfm[:3, 3]  # (3,)
            # tg = tg - smpldata["joints"][0, 0]

            # Joints: Rg · j + tg (apply to all frames)
            j = smpldata["joints"]  # (seq_len, n_joints, 3)
            smpldata["joints"] = torch.einsum("ij,tbj->tbi", Rg, j) + tg

            # Root translation: Rg · t + tg (apply to all frames)
            t = smpldata["trans"]  # (seq_len, 3)
            smpldata["trans"] = torch.einsum("ij,tj->ti", Rg, t) + tg

            # Root orientation: Rg · R_root (apply to root joint only)
            poses_aa = smpldata["poses"]  # (seq_len, n_joints * 3)
            poses_mat = axis_angle_to_matrix(
                einops.rearrange(poses_aa, "t (j c) -> t j c", c=3)
            )  # (seq_len, n_joints, 3, 3)
            poses_mat[:, 0] = Rg @ poses_mat[:, 0]
            smpldata["poses"] = einops.rearrange(
                matrix_to_axis_angle(poses_mat), "t j c -> t (j c)"
            )
        # if tfm is not None:
        #     device = smpldata["joints"].device
        #     tfm = tfm.to(device)  # (4,4)
        #     Rg, tg = tfm[:3, :3], tfm[:3, 3]  # rotation, translation

        #     # (a) joints  →  Rg·j  + tg
        #     j = smpldata["joints"]  # (T, J, 3)
        #     smpldata["joints"] = (
        #         torch.einsum("ij,tbj->tbi", Rg, j) + tg
        #     )

        #     # (b) root translation  →  Rg·t  + tg
        #     t = smpldata["trans"]  # (T, 3)
        #     smpldata["trans"] = torch.einsum("ij,tj->ti", Rg, t) + tg

        #     # (c) root orientation  →  Rg · R_root
        #     #     keep per-frame heading/tilt that you already decoded
        #     poses_aa = smpldata["poses"]  # (T, J*3) axis-angle
        #     poses_mat = axis_angle_to_matrix(
        #         einops.rearrange(
        #             poses_aa, "t (j c) -> t j c", c=3)
        #     )  # (T, J, 3, 3)
        #     poses_mat[:, 0] = torch.matmul(Rg, poses_mat[:, 0])
        #     smpldata["poses"] = einops.rearrange(
        #         matrix_to_axis_angle(poses_mat),
        #         "t j c -> t (j c)")

        return SmplData(**smpldata)

    def group(self, features_dict: Dict[str, Tensor]) -> Tensor:
        """
        Group features from dictionary into a single tensor.
        """
        # Create list of tensors in the order of selected features
        feature_tensors = []

        # Process each selected feature
        for feature in self.features:
            tensor = features_dict[feature.value]

            # Apply reshape if needed
            reshape_func = FEATURE_METADATA[feature]["group_reshape"]
            if reshape_func is not None:
                tensor = reshape_func(tensor)

            feature_tensors.append(tensor)

        # Stack things together
        features, _ = einops.pack(feature_tensors, "k *")
        return features

    def ungroup(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Ungroup features tensor into a dictionary of features.
        """
        # Create result dictionary with properly reshaped features
        feature_dict = {}
        start_idx = 0
        for size, feature_type in zip(self.sizes, self.features):
            tensor = features[:, start_idx : start_idx + size]
            if size == 1:
                tensor = tensor.squeeze(-1)
            start_idx += size
            ungroup_func = FEATURE_METADATA[feature_type]["ungroup_reshape"]
            if ungroup_func is not None:
                tensor = ungroup_func(tensor)
            feature_dict[feature_type.value] = tensor
        return feature_dict

    def get_feature_details(self, feature_type: FeatureType) -> Dict:
        """Get metadata for a specific feature type."""
        return FEATURE_METADATA[feature_type]


def decode_global_trans(trans_mode, ema_alpha, vel_corr_beta, vel, abs_pred):
    if trans_mode == "abs":
        obj_trans = abs_pred

    else:
        base = abs_pred[0]  # (3,)

        # exclusive cumulative sum from velocity
        disps = torch.cat(
            [torch.zeros_like(base).unsqueeze(0), vel[:-1]],
            dim=0,
        )
        vel_int = base + torch.cumsum(disps, dim=0)  # (T, 3)

        if trans_mode == "vel":
            obj_trans = vel_int
        elif trans_mode == "vel_ema":
            obj_trans = (1.0 - ema_alpha) * vel_int + ema_alpha * abs_pred
        elif trans_mode == "vel_corr":
            obj_trans = _integrate_with_correction(vel, abs_pred, beta=vel_corr_beta)
        else:
            raise ValueError(
                f"Unknown trans_mode '{trans_mode}'. "
                "Choose from 'abs', 'vel', or 'vel_ema'."
            )

    return obj_trans


def _integrate_with_correction(vel, abs_pred, beta):
    """
    vel      : (T, 3)  - raw velocities (t → t+1)
    abs_pred : (T, 3)  - absolute positions
    beta     : float   - 0 → no correction, 1 → trust only abs_pred
    """
    base = abs_pred[0]  # (3,)
    pos = [base]  # list for speed

    for t in range(1, vel.size(0)):
        # position after naive integration
        pos_int = pos[-1] + vel[t - 1]

        # proportional correction toward abs_pred[t]
        corr_vel = beta * (abs_pred[t] - pos_int)

        # corrected velocity
        v_star = vel[t - 1] + corr_vel

        # integrate
        pos.append(pos[-1] + v_star)

    return torch.stack(pos, 0)


def axis_angle_to_cont6d(poses: Tensor) -> Tensor:
    """
    (seq_len, 3) -> (seq_len, 6)
    """
    poses_mat = axis_angle_to_matrix(poses)
    cont6d = matrix_to_rotation_6d(poses_mat)
    return cont6d


def cont6d_to_axis_angle(cont6d: Tensor) -> Tensor:
    """
    (seq_len, 6) -> (seq_len, 3)
    """
    poses_mat = rotation_6d_to_matrix(cont6d)
    poses = matrix_to_axis_angle(poses_mat)
    return poses
