from __future__ import annotations
import torch
import einops
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum


from hoidini.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)
from hoidini.datasets.smpldata import SmplData


COMMON_FEATURE_STRINGS = {
    "human": "root_height|root_lin_vel|root_rot_vel|poses|joints",
    "hoi_obj_rel_to_body": "root_height|root_lin_vel|root_rot_vel|poses|joints|poses_obj_r2b|trans_obj_r2b|contact",
    "hoi_obj_rel_to_body_and_hands": "root_height|root_lin_vel|root_rot_vel|poses|joints|poses_obj_r2b|trans_obj_r2b|poses_obj_r2rh|trans_obj_r2rh|poses_obj_r2lh|trans_obj_r2lh|contact",
}


class FeatureType(Enum):
    ROOT_HEIGHT = "root_height"
    ROOT_LIN_VEL = "root_lin_vel"
    ROOT_ROT_VEL = "root_rot_vel"
    POSES = "poses"
    JOINTS = "joints"
    POSES_OBJ_R2B = "poses_obj_r2b"
    TRANS_OBJ_R2B = "trans_obj_r2b"
    POSES_OBJ_R2RH = "poses_obj_r2rh"
    TRANS_OBJ_R2RH = "trans_obj_r2rh"
    POSES_OBJ_R2LH = "poses_obj_r2lh"
    TRANS_OBJ_R2LH = "trans_obj_r2lh"
    CONTACT = "contact"

    @staticmethod
    def from_string(feature_str: str) -> "FeatureType":
        for feature in FeatureType:
            if feature.value == feature_str:
                return feature
        raise ValueError(f"Unknown feature: {feature_str}")

    @staticmethod
    def parse_feature_string(feature_string: str = "hoi") -> List[FeatureType]:
        if feature_string in COMMON_FEATURE_STRINGS:
            feature_string = COMMON_FEATURE_STRINGS[feature_string]
        return [FeatureType.from_string(s.strip()) for s in feature_string.split("|") if s.strip()]


class Feature(ABC):
    def __init__(self, n_pose_joints: int, n_contact_verts: Optional[int] = None):
        self.n_pose_joints = n_pose_joints
        self.n_contact_verts = n_contact_verts

    @abstractmethod
    @property
    def feature_type(self) -> FeatureType:
        pass
    
    @abstractmethod
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        pass

    @abstractmethod
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        pass

    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        pass

    @abstractmethod
    def size_func(self) -> int:
        pass
    
    def group_reshape(self, tensor: Tensor) -> Tensor:
        return tensor
    
    def ungroup_reshape(self, tensor: Tensor) -> Tensor:
        return tensor
    
    def encode(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        for feature in self.required_features_for_encoding:
            if feature not in metadata:
                metadata[feature] = FEATURE_MAP[feature](self.n_pose_joints, self.n_contact_verts).encode(smpldata, metadata)
        if self.feature_type.value not in metadata:
            metadata[self.feature_type.value] = self.encode_impl(smpldata, metadata)
        return metadata[self.feature_type.value]

    def decode(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        for feature in self.required_features_for_decoding:
            if feature not in metadata:
                metadata[feature] = FEATURE_MAP[feature](self.n_pose_joints, self.n_contact_verts).decode(feature, metadata)
        if self.feature_type.value not in metadata:
            metadata[self.feature_type.value] = self.decode_impl(feature, metadata)
        return metadata[self.feature_type.value]


class RootHeightFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.ROOT_HEIGHT
    
    def size_func(self) -> int:
        return 1
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return []
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        # Extract joints data
        joints = smpldata["joints"].clone()
        
        # First remove the ground
        ground = joints[:, :, 2].min()
        joints[:, :, 2] -= ground
        
        # Get root height along gravity axis
        root_grav_axis = joints[:, 0, 2].clone()
        
        # Store processed joints in metadata for other features
        metadata["processed_joints"] = joints
        metadata["ground"] = ground
        
        return root_grav_axis
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        metadata["root_grav_axis"] = feature
        return {}


class RootLinVelFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.ROOT_LIN_VEL
    
    def size_func(self) -> int:
        return 2
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_HEIGHT, FeatureType.ROOT_ROT_VEL]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_ROT_VEL]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        joints = metadata["processed_joints"]
        rotZ = metadata["rotZ"]
        
        # Get trajectory (XY of root joint)
        trajectory = joints[:, 0, :2].clone()
        metadata["trajectory"] = trajectory
        
        # Calculate trajectory velocity
        vel_trajectory = torch.diff(trajectory, dim=0)
        # Repeat the last acceleration for future velocity
        last_acceleration = vel_trajectory[-1] - vel_trajectory[-2]
        future_velocity = vel_trajectory[-1] + last_acceleration
        vel_trajectory = torch.cat((vel_trajectory, future_velocity[None]), dim=0)
        metadata["vel_trajectory"] = vel_trajectory
        
        # Rotate to local coordinate system
        vel_trajectory_local = torch.einsum("tkj,tk->tj", rotZ[:, :2, :2], vel_trajectory)
        
        return vel_trajectory_local
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        vel_trajectory_local = feature
        metadata["vel_trajectory_local"] = vel_trajectory_local
        
        # Get rotation matrix from metadata
        rotZ = metadata["rotZ"]
        
        # Rotate trajectory from local to global
        vel_trajectory = torch.einsum("bjk,bk->bj", rotZ[:, :2, :2], vel_trajectory_local)
        metadata["vel_trajectory"] = vel_trajectory
            
        # Integrate to get trajectory
        trajectory = torch.cumsum(vel_trajectory[..., :-1, :], dim=-2)
        trajectory = torch.cat((0 * trajectory[..., [0], :], trajectory), dim=-2)
        metadata["trajectory"] = trajectory
        
        # This feature doesn't directly map to a SmplData field
        return {}


class RootRotVelFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.ROOT_ROT_VEL
    
    def size_func(self) -> int:
        return 1
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return []
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        # Calculate rotation matrices from poses
        poses = smpldata["poses"].clone()
        poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
        poses_mat = axis_angle_to_matrix(poses)
        global_orient = poses_mat[:, 0]
        global_euler = matrix_to_euler_angles(global_orient, "ZYX")
        rotZ_angle = global_euler[:, 0]
        rotZ = axis_angle_rotation("Z", rotZ_angle)
        
        # Store in metadata for other features
        metadata["rotZ"] = rotZ
        metadata["global_orient"] = global_orient
        metadata["global_euler"] = global_euler
            
        # Calculate rotation velocity
        vel_rotZ = rotZ[1:] @ rotZ.transpose(1, 2)[:-1]
        # Repeat last acceleration for future velocity
        last_acc_rotZ = vel_rotZ[-1] @ vel_rotZ.transpose(1, 2)[-2]
        future_vel_rotZ = vel_rotZ[-1] @ last_acc_rotZ
        vel_rotZ = torch.cat((vel_rotZ, future_vel_rotZ[None]), dim=-3)
        vel_angles = matrix_to_axis_angle(vel_rotZ)[:, 2]
        
        return vel_angles
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        vel_angles = feature
        metadata["vel_angles"] = vel_angles
        
        # Integrate angles
        angles = torch.cumsum(vel_angles[:-1], dim=0)
        angles = torch.cat((0 * angles[[0]], angles), dim=0)
        
        # Construct rotation matrix
        rotZ = axis_angle_rotation("Z", angles)
        metadata["rotZ"] = rotZ
        metadata["angles"] = angles
        
        return {self.feature_type.value: vel_angles}


class PosesFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.POSES
    
    def size_func(self) -> int:
        return self.n_pose_joints * 6
    
    def group_reshape(self, tensor: Tensor) -> Tensor:
        return einops.rearrange(tensor, "k l t -> k (l t)")
    
    def ungroup_reshape(self, tensor: Tensor) -> Tensor:
        return einops.rearrange(tensor, "k (l t) -> k l t", t=6)
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_ROT_VEL]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_ROT_VEL]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        # Access rotation data from metadata
        global_euler = metadata["global_euler"]
        
        # Get poses
        poses = smpldata["poses"].clone()
        poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
        poses_mat = axis_angle_to_matrix(poses)
        metadata["poses_mat"] = poses_mat
        
        # Decompose global orientation
        _, rotY_angle, rotX_angle = torch.unbind(global_euler, -1)
        
        # Construct rotation matrices for each axis
        rotY = axis_angle_rotation("Y", rotY_angle)
        rotX = axis_angle_rotation("X", rotX_angle)
        
        # Construct local global pose (without final Z rotation)
        global_orient_local = rotY @ rotX
        metadata["global_orient_local"] = global_orient_local
        
        # Replace global orientation with local version
        poses_mat_local = torch.cat((global_orient_local[:, None], poses_mat[:, 1:]), dim=1)
        
        # Convert to 6D representation
        poses_local = matrix_to_rotation_6d(poses_mat_local)
        metadata["poses_mat_local"] = poses_mat_local
        
        return poses_local
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        poses_local = feature
        
        # Convert 6D representation back to rotation matrices
        poses_mat_local = rotation_6d_to_matrix(poses_local)
        metadata["poses_mat_local"] = poses_mat_local
        
        # Get global orientation without Z rotation
        global_orient_local = poses_mat_local[:, 0]
        metadata["global_orient_local"] = global_orient_local
        
        # Get rotation matrix
        rotZ = metadata["rotZ"]
        
        # Extract Y and X components from global_orient_local
        global_euler_local = matrix_to_euler_angles(global_orient_local, "ZYX")
        _, rotY_angle, rotX_angle = torch.unbind(global_euler_local, -1)
        rotY = axis_angle_rotation("Y", rotY_angle)
        rotX = axis_angle_rotation("X", rotX_angle)
        
        # Reconstruct full global orientation
        global_orient = rotZ @ rotY @ rotX
        
        # Put back into full pose matrix
        poses_mat = torch.cat((global_orient[:, None], poses_mat_local[:, 1:]), dim=1)
        
        # Convert to axis-angle representation
        poses = matrix_to_axis_angle(poses_mat)
        
        # Flatten
        poses = einops.rearrange(poses, "k l t -> k (l t)")
        
        # Store in metadata
        metadata["poses_mat"] = poses_mat
        
        # This maps directly to poses in SmplData
        return {"poses": poses}


class JointsFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.JOINTS
    
    def size_func(self) -> int:
        return (self.n_pose_joints - 1) * 3
    
    def group_reshape(self, tensor: Tensor) -> Tensor:
        return einops.rearrange(tensor, "k l t -> k (l t)")
    
    def ungroup_reshape(self, tensor: Tensor) -> Tensor:
        return einops.rearrange(tensor, "k (l t) -> k l t", t=3)
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_HEIGHT, FeatureType.ROOT_ROT_VEL]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_HEIGHT, FeatureType.ROOT_ROT_VEL, FeatureType.ROOT_LIN_VEL]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        # Access processed data from metadata
        joints = metadata["processed_joints"]
        rotZ = metadata["rotZ"]
        
        # Get trajectory
        trajectory = joints[:, 0, :2].clone()
        metadata["trajectory"] = trajectory
        
        # Remove trajectory from joints XY
        joints[:, :, [0, 1]] -= trajectory[:, None, :]
        
        # Also normalize Z coordinate to pelvis height
        joints[:, :, 2] -= joints[:, [0], 2]
        
        # Verify pelvis is at origin
        assert torch.allclose(joints[:, 0], torch.zeros_like(joints[:, 0]))
        
        # Remove pelvis (already encoded in height and trajectory)
        joints = joints[:, 1:]
        
        # Rotate XY to local coordinate frame
        joints_xy_local = torch.einsum("tkj,tlk->tlj", rotZ[:, :2, :2], joints[:, :, [0, 1]])
        joints_local = torch.cat([joints_xy_local, joints[:, :, 2:3]], dim=-1)
        
        return joints_local
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        joints_local = feature
        
        # Get rotation matrix and other data
        rotZ = metadata["rotZ"]
        root_grav_axis = metadata["root_grav_axis"]
        trajectory = metadata["trajectory"]
        
        # Rotate joints back to global frame
        joints_xy = torch.einsum("bjk,blk->blj", rotZ[:, :2, :2], joints_local[..., [0, 1]])
        joints = torch.cat([joints_xy, joints_local[..., 2:3]], dim=-1)
        
        # Add back the pelvis (zero)
        joints = torch.cat((0 * joints[:, [0]], joints), dim=1)
        
        # Add back the height
        joints[:, :, 2] += root_grav_axis[:, None]
        
        # Add back the trajectory
        joints[:, :, [0, 1]] += trajectory[:, None, :]
        
        # Store for other features
        metadata["joints_out"] = joints
        
        # This maps directly to joints in SmplData
        return {"joints": joints}


class PosesObjR2BFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.POSES_OBJ_R2B
    
    def size_func(self) -> int:
        return 6
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_ROT_VEL]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_ROT_VEL]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        if "poses_obj" not in smpldata:
            raise ValueError("Object poses not found in input data")
        
        # Access rotation matrix from metadata
        rotZ = metadata["rotZ"]
        
        # Get object poses
        pose_obj = smpldata["poses_obj"].clone()
        
        # Convert to rotation matrix
        pose_obj_mat = axis_angle_to_matrix(pose_obj.view(-1, 1, 3))[:, 0]
        
        # Make relative to root rotation
        pose_obj_mat = rotZ.transpose(1, 2) @ pose_obj_mat
        
        # Convert to 6D representation
        pose_obj = matrix_to_rotation_6d(pose_obj_mat)
        
        return pose_obj
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        pose_obj = feature
        
        # Access rotation matrix from metadata
        rotZ = metadata["rotZ"]
        
        # Convert back to rotation matrix
        pose_obj_mat = rotation_6d_to_matrix(pose_obj)
        
        # Transform back to global frame
        pose_obj_mat = rotZ @ pose_obj_mat
        
        # Convert to axis-angle
        pose_obj = matrix_to_axis_angle(pose_obj_mat)
        
        metadata["poses_obj"] = pose_obj
        
        # This maps directly to poses_obj in SmplData
        return {"poses_obj": pose_obj}


class TransObjR2BFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.TRANS_OBJ_R2B
    
    def size_func(self) -> int:
        return 3
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_HEIGHT, FeatureType.ROOT_ROT_VEL]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.ROOT_HEIGHT, FeatureType.ROOT_ROT_VEL, FeatureType.ROOT_LIN_VEL]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        if "trans_obj" not in smpldata:
            raise ValueError("Object translation not found in input data")
        
        # Get object translation
        trans_obj = smpldata["trans_obj"].clone()
        
        # Get trajectory and root height from metadata
        trajectory = metadata["trajectory"]
        root_grav_axis = metadata["root_grav_axis"]
        rotZ = metadata["rotZ"]
        
        # Make translation relative to root
        trans_obj[:, :2] -= trajectory  # XY relative to trajectory
        trans_obj[:, 2] -= root_grav_axis  # Z relative to root height
        
        # Rotate to be in root-aligned frame
        trans_obj_xy = torch.einsum("tkj,tk->tj", rotZ[:, :2, :2], trans_obj[:, :2])
        trans_obj = torch.cat([trans_obj_xy, trans_obj[:, 2:3]], dim=-1)
        
        return trans_obj
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        trans_obj = feature
        
        # Get rotation matrix and other data
        rotZ = metadata["rotZ"]
        root_grav_axis = metadata["root_grav_axis"]
        trajectory = metadata["trajectory"]
        
        # Transform back to global coordinates
        trans_obj_xy = torch.einsum("bjk,bk->bj", rotZ[:, :2, :2], trans_obj[:, :2])
        trans_obj = torch.cat([trans_obj_xy, trans_obj[:, 2:3]], dim=-1)
        
        # Add back root height and trajectory
        trans_obj[:, 2] += root_grav_axis
        trans_obj[:, :2] += trajectory
        
        # This maps directly to trans_obj in SmplData
        return {"trans_obj": trans_obj}


class PosesObjR2RHFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.POSES_OBJ_R2RH
    
    def size_func(self) -> int:
        return 6
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        if "poses_obj" not in smpldata:
            raise ValueError("Object poses not found in input data")
        
        # Assuming right hand is at a specific joint index (e.g. 21)
        right_hand_idx = 21
        
        # Get object pose
        pose_obj = smpldata["poses_obj"].clone()
        pose_obj_mat = axis_angle_to_matrix(pose_obj.view(-1, 1, 3))[:, 0]
        
        # Get joints data
        joints = smpldata["joints"].clone()
        
        # Calculate right hand orientation
        right_wrist_idx = right_hand_idx
        right_elbow_idx = right_hand_idx - 1
        
        # Compute hand direction vector
        hand_dir = joints[:, right_wrist_idx] - joints[:, right_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        # Create hand orientation matrix
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        rh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Make object orientation relative to right hand
        pose_obj_r2rh_mat = rh_rot_mat.transpose(-1, -2) @ pose_obj_mat
        
        # Convert to 6D representation
        pose_obj_r2rh = matrix_to_rotation_6d(pose_obj_r2rh_mat)
        
        return pose_obj_r2rh
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        pose_obj_r2rh = feature
        
        # Convert 6D representation back to rotation matrix
        pose_obj_r2rh_mat = rotation_6d_to_matrix(pose_obj_r2rh)
        
        # Assuming right hand is at joint index 21
        right_hand_idx = 21
        
        # Get joint data from metadata
        joints = metadata["joints_out"]
        
        # Compute hand orientation
        right_wrist_idx = right_hand_idx
        right_elbow_idx = right_hand_idx - 1
        
        hand_dir = joints[:, right_wrist_idx] - joints[:, right_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        rh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Transform back to global coordinates
        pose_obj_mat = rh_rot_mat @ pose_obj_r2rh_mat
        
        # Convert to axis-angle
        pose_obj = matrix_to_axis_angle(pose_obj_mat)
        
        metadata["poses_obj"] = pose_obj
        
        return {self.feature_type.value: pose_obj_r2rh}


class TransObjR2RHFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.TRANS_OBJ_R2RH
    
    def size_func(self) -> int:
        return 3
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        if "trans_obj" not in smpldata:
            raise ValueError("Object translation not found in input data")
        
        # Assuming right hand is at joint index 21
        right_hand_idx = 21
        
        # Get object translation
        trans_obj = smpldata["trans_obj"].clone()
        
        # Get joint data
        joints = smpldata["joints"].clone()
        
        # Get right hand position
        right_hand_pos = joints[:, right_hand_idx]
        
        # Make translation relative to right hand
        trans_obj_r2rh = trans_obj - right_hand_pos
        
        # Get right hand orientation for local coordinate transform
        right_wrist_idx = right_hand_idx
        right_elbow_idx = right_hand_idx - 1
        
        hand_dir = joints[:, right_wrist_idx] - joints[:, right_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        rh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Transform to hand-aligned coordinate system
        trans_obj_r2rh_local = torch.einsum("bij,bj->bi", rh_rot_mat.transpose(-1, -2), trans_obj_r2rh)
        
        return trans_obj_r2rh_local
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        trans_obj_r2rh_local = feature
        
        # Assuming right hand is at joint index 21
        right_hand_idx = 21
        
        # Get joint data from metadata
        joints = metadata["joints_out"]
        
        # Get right hand position
        right_hand_pos = joints[:, right_hand_idx]
        
        # Compute hand orientation as done in encode_impl
        right_wrist_idx = right_hand_idx
        right_elbow_idx = right_hand_idx - 1
        
        hand_dir = joints[:, right_wrist_idx] - joints[:, right_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        rh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Transform back to global coordinate system
        trans_obj_r2rh = torch.einsum("bij,bj->bi", rh_rot_mat, trans_obj_r2rh_local)
        
        # Add right hand position to get global position
        trans_obj = trans_obj_r2rh + right_hand_pos
        
        # Add to metadata for other features to use
        metadata["trans_obj"] = trans_obj
        
        return {self.feature_type.value: trans_obj_r2rh_local}


class PosesObjR2LHFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.POSES_OBJ_R2LH
    
    def size_func(self) -> int:
        return 6
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        if "poses_obj" not in smpldata:
            raise ValueError("Object poses not found in input data")
        
        # Assuming left hand is at joint index 20
        left_hand_idx = 20
        
        # Get object pose
        pose_obj = smpldata["poses_obj"].clone()
        pose_obj_mat = axis_angle_to_matrix(pose_obj.view(-1, 1, 3))[:, 0]
        
        # Get joints data to extract left hand orientation
        joints = smpldata["joints"].clone()
        
        # Calculate left hand orientation (simplified approach)
        # In a real implementation, you would compute the hand's orientation from the joint hierarchy
        left_wrist_idx = left_hand_idx
        left_elbow_idx = left_hand_idx - 1
        
        # Compute hand direction vector
        hand_dir = joints[:, left_wrist_idx] - joints[:, left_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        # Create simplified hand orientation matrix (approximate)
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        lh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Make object orientation relative to left hand
        pose_obj_r2lh_mat = lh_rot_mat.transpose(-1, -2) @ pose_obj_mat
        
        # Convert to 6D representation
        pose_obj_r2lh = matrix_to_rotation_6d(pose_obj_r2lh_mat)
        
        return pose_obj_r2lh
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        pose_obj_r2lh = feature
        
        # Convert 6D representation back to rotation matrix
        pose_obj_r2lh_mat = rotation_6d_to_matrix(pose_obj_r2lh)
        
        # Assuming left hand is at joint index 20
        left_hand_idx = 20
        
        # Get joint data from metadata
        joints = metadata["joints_out"]
        
        # Compute hand orientation as done in encode_impl
        left_wrist_idx = left_hand_idx
        left_elbow_idx = left_hand_idx - 1
        
        hand_dir = joints[:, left_wrist_idx] - joints[:, left_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        lh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Transform back to global coordinates
        pose_obj_mat = lh_rot_mat @ pose_obj_r2lh_mat
        
        # Convert to axis-angle
        pose_obj = matrix_to_axis_angle(pose_obj_mat)
        
        # Add to metadata for other features to use
        metadata["poses_obj"] = pose_obj
        
        return {self.feature_type.value: pose_obj_r2lh}


class TransObjR2LHFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.TRANS_OBJ_R2LH
    
    def size_func(self) -> int:
        return 3
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    @property
    def required_features_for_decoding(self) -> List[FeatureType]:
        return [FeatureType.JOINTS]
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        if "trans_obj" not in smpldata:
            raise ValueError("Object translation not found in input data")
        
        # Assuming left hand is at joint index 20
        left_hand_idx = 20
        
        # Get object translation
        trans_obj = smpldata["trans_obj"].clone()
        
        # Get joint data
        joints = smpldata["joints"].clone()
        
        # Get left hand position
        left_hand_pos = joints[:, left_hand_idx]
        
        # Make translation relative to left hand
        trans_obj_r2lh = trans_obj - left_hand_pos
        
        # Get left hand orientation for local coordinate transform
        left_wrist_idx = left_hand_idx
        left_elbow_idx = left_hand_idx - 1
        
        hand_dir = joints[:, left_wrist_idx] - joints[:, left_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        lh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Transform to hand-aligned coordinate system
        trans_obj_r2lh_local = torch.einsum("bij,bj->bi", lh_rot_mat.transpose(-1, -2), trans_obj_r2lh)
        
        return trans_obj_r2lh_local
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        trans_obj_r2lh_local = feature
        
        # Assuming left hand is at joint index 20
        left_hand_idx = 20
        
        # Get joint data from metadata
        joints = metadata["joints_out"]
        
        # Get left hand position
        left_hand_pos = joints[:, left_hand_idx]
        
        # Compute hand orientation as done in encode_impl
        left_wrist_idx = left_hand_idx
        left_elbow_idx = left_hand_idx - 1
        
        hand_dir = joints[:, left_wrist_idx] - joints[:, left_elbow_idx]
        hand_dir = hand_dir / (torch.norm(hand_dir, dim=-1, keepdim=True) + 1e-8)
        
        z_axis = hand_dir
        x_axis = torch.cross(z_axis, torch.tensor([0.0, 0.0, 1.0], device=z_axis.device))
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-8)
        y_axis = torch.cross(z_axis, x_axis)
        
        lh_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        
        # Transform back to global coordinate system
        trans_obj_r2lh = torch.einsum("bij,bj->bi", lh_rot_mat, trans_obj_r2lh_local)
        
        # Add left hand position to get global position
        trans_obj = trans_obj_r2lh + left_hand_pos
        
        # Add to metadata for other features to use
        metadata["trans_obj"] = trans_obj
        
        return {self.feature_type.value: trans_obj_r2lh_local}


class ContactFeature(Feature):
    @property
    def feature_type(self) -> FeatureType:
        return FeatureType.CONTACT
    
    def size_func(self) -> int:
        return self.n_contact_verts
    
    @property
    def required_features_for_encoding(self) -> List[FeatureType]:
        return []
    
    def encode_impl(self, smpldata: Dict[str, Tensor], metadata: Dict[str, Any]) -> Tensor:
        if "contact" not in smpldata:
            raise ValueError("Contact data not found in input data")
        
        # Contact data is used as-is
        contact = smpldata["contact"].clone()
        
        return contact
    
    def decode_impl(self, feature: Tensor, metadata: Dict[str, Any]) -> Dict[str, Tensor]:
        # This maps directly to contact in SmplData
        return {"contact": feature}


FEATURE_MAP = {
    FeatureType.ROOT_HEIGHT: RootHeightFeature,
    FeatureType.ROOT_LIN_VEL: RootLinVelFeature,
    FeatureType.ROOT_ROT_VEL: RootRotVelFeature,
    FeatureType.POSES: PosesFeature,
    FeatureType.JOINTS: JointsFeature,
    FeatureType.POSES_OBJ_R2B: PosesObjR2BFeature,
    FeatureType.TRANS_OBJ_R2B: TransObjR2BFeature,
    FeatureType.POSES_OBJ_R2RH: PosesObjR2RHFeature,
    FeatureType.TRANS_OBJ_R2RH: TransObjR2RHFeature,
    FeatureType.POSES_OBJ_R2LH: PosesObjR2LHFeature,
    FeatureType.TRANS_OBJ_R2LH: TransObjR2LHFeature,
    FeatureType.CONTACT: ContactFeature,
}


class SMPLFeatureProcessor:
    def __init__(
            self,
            n_pose_joints: int = 52,
            n_contact_verts: Optional[int] = None,
            features_string: Optional[str] = None
            ):
        self.n_pose_joints = n_pose_joints
        self.n_contact_verts = n_contact_verts
        self.features_string = features_string

        self.feature_types = FeatureType.parse_feature_string(self.features_string)
        
        # Initialize feature objects
        self.features: List[Feature] = []
        for feature_type in self.feature_types:
            FeatureClass = FEATURE_MAP[feature_type]
            self.features.append(FeatureClass(n_pose_joints, n_contact_verts))

        self.sizes = [feature.size_func() for feature in self.features]

    def encode(self, smpldata: SmplData) -> Tensor:
        smpldata_dict = smpldata.to_dict()
        metadata = {}
        features_dict = {}
        for feature in self.features:
            feature_tensor = feature.encode(smpldata_dict, metadata)
            features_dict[feature.feature_type.value] = feature_tensor
        
        features = self.group(features_dict)
        return features

    def decode(self, features: Tensor) -> SmplData:
        feature_dict = self.ungroup(features)
        metadata = {}
        smpldata = {}
        for feature in self.features:
            feature_tensor = feature_dict[feature.feature_type.value]
            metadata[feature.feature_type.value] = feature.decode(feature_tensor, metadata)
            smpldata.update(metadata[feature.feature_type.value])
        return SmplData(**smpldata)

    def group(self, features_dict: Dict[str, Tensor]) -> Tensor:
        feature_tensors = []
        for feature in self.features:
            tensor = features_dict[feature.feature_type.value]
            tensor = feature.group_reshape(tensor)
            feature_tensors.append(tensor)
        features, _ = einops.pack(feature_tensors, "k *")
        return features

    def ungroup(self, features: Tensor) -> Dict[str, Tensor]:
        feature_dict = {}
        start_idx = 0
        for size, feature in zip(self.sizes, self.features):
            tensor = features[:, start_idx:start_idx + size]
            if size == 1:
                tensor = tensor.squeeze(-1)
            start_idx += size
            tensor = feature.ungroup_reshape(tensor)
            feature_dict[feature.feature_type.value] = tensor
        return feature_dict
