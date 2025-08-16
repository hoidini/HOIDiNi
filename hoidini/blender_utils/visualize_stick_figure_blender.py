import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Union
from multiprocessing import current_process
import pickle
import random
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import torch

from hoidini.blender_utils.axis_xyz import create_xyz_axes
from hoidini.blender_utils.general_blender_utils import (
    CollectionManager,
    blend_scp_and_run,
)
from hoidini.blender_utils.visualize_mesh_figure_blender import (
    animate_mesh,
    get_smpl_template,
)
from hoidini.skeletons.smpl_24 import BONES_24_INDS
from hoidini.skeletons.smplx_52 import BONE_52_52_INDS

if current_process().name == "MainProcess":  # enables multiprocessing
    import bpy


def zup2yup(motion_xyz):
    """
    Convert Y-up coordinates to Z-up coordinates using a 90-degree rotation around the X-axis.

    Supports both NumPy and PyTorch tensors.

    Args:
        motion_xyz (np.ndarray or torch.Tensor): 3D points of shape (..., 3).

    Returns:
        np.ndarray or torch.Tensor: Transformed coordinates, same type as input.
    """
    is_torch = isinstance(motion_xyz, torch.Tensor)

    if is_torch:
        device = motion_xyz.device  # Store device
        dtype = motion_xyz.dtype  # Store dtype
        motion_xyz_np = motion_xyz.cpu().numpy()  # Convert to NumPy
    else:
        motion_xyz_np = np.asarray(motion_xyz)  # Ensure NumPy array

    # Apply 90-degree rotation around the X-axis
    orig_shape = motion_xyz_np.shape
    rotated_np = R.from_euler("x", -90, degrees=True).apply(
        motion_xyz_np.reshape(-1, 3)
    )

    # Convert back to original type
    if is_torch:
        return torch.tensor(rotated_np, dtype=dtype, device=device).view(orig_shape)
    else:
        return rotated_np.reshape(orig_shape)


def yup2zup(motion_xyz):
    """
    Convert Y-up coordinates to Z-up coordinates using a 90-degree rotation around the X-axis.

    Supports both NumPy and PyTorch tensors.

    Args:
        motion_xyz (np.ndarray or torch.Tensor): 3D points of shape (..., 3).

    Returns:
        np.ndarray or torch.Tensor: Transformed coordinates, same type as input.
    """
    is_torch = isinstance(motion_xyz, torch.Tensor)

    if is_torch:
        device = motion_xyz.device  # Store device
        dtype = motion_xyz.dtype  # Store dtype
        motion_xyz_np = motion_xyz.cpu().numpy()  # Convert to NumPy
    else:
        motion_xyz_np = np.asarray(motion_xyz)  # Ensure NumPy array

    # Apply 90-degree rotation around the X-axis
    orig_shape = motion_xyz_np.shape
    rotated_np = R.from_euler("x", 90, degrees=True).apply(motion_xyz_np.reshape(-1, 3))

    # Convert back to original type
    if is_torch:
        return torch.tensor(rotated_np, dtype=dtype, device=device).view(orig_shape)
    else:
        return rotated_np.reshape(orig_shape)


class SkeletonSizeMap(ABC):
    # TODO make this class control colors as well
    @abstractmethod
    def get_joint_size(self, joint_ind):
        pass

    @abstractmethod
    def get_bone_size(self, head, tail):
        pass


class DefaultSkeletonSizeMap(SkeletonSizeMap):
    def __init__(self, finger_indices_start=23):
        super().__init__()
        self.finger_indices_start = finger_indices_start

    def get_joint_size(self, joint_ind):
        if joint_ind >= self.finger_indices_start:
            return 0.01
        else:
            return 0.03

    def get_bone_size(self, head_ind, tail_ind):
        if tail_ind >= self.finger_indices_start:
            return 0.007
        else:
            return 0.02


def set_duration(*frames):
    if len(frames) == 1:
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = frames[0] - 1
    elif len(frames) == 2:
        bpy.context.scene.frame_start = frames[0]
        bpy.context.scene.frame_end = frames[1]


def add_chessboard_floor():
    # Create a new plane
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, location=(0, 0, 0))
    # Scale the plane (optional)
    bpy.context.object.scale = (10, 10, 10)  # Scale by 10x in each dimension
    # Create new material with a checker texture
    material = bpy.data.materials.new(name="Checkered_Material")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    checker_tex = material.node_tree.nodes.new("ShaderNodeTexChecker")
    checker_tex.inputs["Scale"].default_value = 10  # Adjust scale of the checkers
    # Connect checker texture to BSDF base color
    material.node_tree.links.new(
        bsdf.inputs["Base Color"], checker_tex.outputs["Color"]
    )
    # Assign material to plane
    plane = bpy.context.object
    plane.data.materials.append(material)


def add_lights():
    CollectionManager.create_collection("Lights")

    def add_light(location, name=None):
        object_hook = ObjectHook()
        bpy.ops.object.light_add(
            type="SUN", align="WORLD", location=location, scale=(1, 1, 1)
        )
        object_hook.name_and_get(name)

    locs = [(-5, -5, 5), (0, 0, 10), (5, 5, 5), (4, 4, 15)]
    for i, loc in enumerate(locs):
        name = f"light_{i:03d}"
        add_light(loc, name)
        CollectionManager.add_object_to_collection(name, "Lights")


def render(output_path):
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # ('BLENDER_EEVEE_NEXT', 'BLENDER_WORKBENCH', 'CYCLES')
    render_settings = bpy.context.scene.render
    render_settings.filepath = output_path
    render_settings.ffmpeg.format = "MPEG4"  # some problems with cs.tau ffmpeg
    render_settings.ffmpeg.codec = "H264"  # some problems with cs.tau ffmpeg
    # render_settings.ffmpeg.constant_rate_factor = 'HIGH'
    # renderender_settingsr.ffmpeg.ffmpeg_preset = 'MEDIUM'
    bpy.ops.render.render(animation=True)
    print("Rendering Done, saved to", output_path)


def set_render_params(res=1024, fps=20):
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    bpy.context.scene.render.ffmpeg.format = "MPEG4"
    bpy.context.scene.render.resolution_x = res
    bpy.context.scene.render.resolution_y = res
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.render.fps = fps


def delete_all_data():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    for coll in bpy.data.collections:
        if coll != bpy.data.collections["Collection"]:
            bpy.data.collections.remove(coll)

    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)


def save_blender_file(path="/tmp/debug.blend"):
    print(f"{10 * '*'} saved {path}")
    bpy.ops.wm.save_as_mainfile(filepath=path)


def reset_blender():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_rotation_matrix_between_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    # get a rotation matrix from v0 to v1
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    cos_t = np.dot(v0, v1)
    sin_t = np.linalg.norm(np.cross(v0, v1))

    u = v0
    v = v1 - np.dot(v0, v1) * v0
    v = v / np.linalg.norm(v)
    w = np.cross(v0, v1)
    w = w / np.linalg.norm(w)

    # change of basis matrix
    C = np.array([u, v, w])

    # rotation matrix in new basis
    R_uvw = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    # full rotation matrix
    R = C.T @ R_uvw @ C
    return R


class ObjectHook:
    """
    Used to get a newly added object in blender
    """

    def __init__(self) -> None:
        self.names_before = [e.name for e in bpy.data.objects]

    def name_and_get(self, new_name=None):
        names_after = [e.name for e in bpy.data.objects]
        delta_names = set(names_after) - set(self.names_before)
        assert len(delta_names) == 1
        name = list(delta_names)[0]
        obj = bpy.data.objects[name]
        if new_name is not None:
            obj.name = new_name
        return obj


# class CollectionManager:
#     @staticmethod
#     def create_collection(collection_name):
#         new_collection = bpy.data.collections.new(collection_name)
#         bpy.context.scene.collection.children.link(new_collection)

#     @staticmethod
#     def add_object_to_collection(object_name, collection_name):
#         object_to_add = bpy.data.objects.get(object_name)
#         for other_collection in object_to_add.users_collection:
#             other_collection.objects.unlink(object_to_add)
#         collection = bpy.data.collections[collection_name]
#         collection.objects.link(object_to_add)


class ActiveCamera:
    def __init__(self) -> None:
        """
        Work in progress, not really active yet
        """
        location = np.array([-4.31, -2.35, 2.16])
        rotation = np.array([1.3, 0.01, -1.04])
        bpy.ops.object.camera_add(
            enter_editmode=False,
            align="VIEW",
            location=location,
            rotation=rotation,
            scale=(1, 1, 1),
        )
        camera = bpy.data.objects["Camera"]
        bpy.data.scenes["Scene"].camera = camera


def add_camera():
    ActiveCamera()


def create_spheres_animation(
    vertices, point_size=0.012, name="", max_points=np.inf, color=None
):
    """
    vertices: [sequence_len, vertices_id, 3]
    name to be used in case of multiple
    """
    num_frames, num_points, _ = vertices.shape

    # Randomly select a subset of 400 points
    if num_points > max_points:
        selected_indices = random.sample(range(num_points), max_points)
        vertices = vertices[:, selected_indices, :]
        num_points = max_points

    # Create spheres for each point

    # if color
    # # Add material
    # mat = bpy.data.materials.new(name=f"SphereMaterial_{1}")
    # mat.diffuse_color = (*color, 1)
    # sphere.active_material = mat

    spheres = []
    for i in tqdm(range(num_points), desc="Place spheres"):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=point_size, location=vertices[0, i])
        sphere = bpy.context.object
        sphere.name = f"Vert_Sphere_{i}_{name}"
        spheres.append(sphere)

    # Animate spheres
    for frame in tqdm(range(num_frames)):
        bpy.context.scene.frame_set(frame)
        for i, sphere in enumerate(spheres):
            sphere.location = vertices[frame, i]
            sphere.keyframe_insert(data_path="location", frame=frame)


class CylinderBone:
    def __init__(
        self,
        head_ind: int = None,
        tail_ind: int = None,
        unq_name: str = None,
        radius=0.03,
    ) -> None:
        """
        Use name in case of multiple skeletons with the same bones
        """
        self.head_name = Names.get_joint_name(head_ind, unq_name)
        self.tail_name = Names.get_joint_name(tail_ind, unq_name)
        self.bone_name = Names.get_bone_name(head_ind, tail_ind, unq_name)
        self.unq_name = unq_name

        self.location = None
        self.rotation = None
        self.head_loc = None
        self.tail_loc = None
        self.radius = radius
        bpy.ops.mesh.primitive_cylinder_add(
            radius=self.radius,
            depth=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        cyl = bpy.context.object
        cyl.name = self.bone_name

    def set_head_and_tail_locations_per_frame(
        self, head_loc: np.ndarray, tail_loc: np.ndarray, frame=0
    ):
        cyl = bpy.data.objects[self.bone_name]
        v_src = np.array([0, 0, 1])  # default cylinder direction
        v_tgt = head_loc - tail_loc
        v_tgt = v_tgt / np.linalg.norm(v_tgt)
        cyl.rotation_mode = "QUATERNION"
        ix, iy, iz, w = tuple(
            R.from_matrix(get_rotation_matrix_between_vectors(v_src, v_tgt)).as_quat()
        )
        cyl.rotation_quaternion[0] = w
        cyl.rotation_quaternion[1] = ix
        cyl.rotation_quaternion[2] = iy
        cyl.rotation_quaternion[3] = iz
        cyl.keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # length
        z_dim = float(np.linalg.norm(head_loc - tail_loc))
        cyl.scale.z = z_dim
        cyl.keyframe_insert(data_path="scale", frame=frame)

        # location
        self.location = (head_loc + tail_loc) / 2
        cyl.location = self.location
        cyl.keyframe_insert(data_path="location", frame=frame)
        # if self.head_name is not None:

        joint_head = bpy.data.objects[self.head_name]
        joint_tail = bpy.data.objects[self.tail_name]
        joint_head.location = head_loc
        joint_head.keyframe_insert(data_path="location", frame=frame)
        joint_tail.location = tail_loc
        joint_tail.keyframe_insert(data_path="location", frame=frame)


class Names:
    @staticmethod
    def get_joint_name(joint_ind, unq_name):
        return f"{joint_ind}" if unq_name is None else f"{joint_ind}_{unq_name}"

    @staticmethod
    def get_bone_name(head, tail, unq_name):
        return (
            f"cyl_{head}_{tail}"
            if unq_name is None
            else f"cyl_{head}_{tail}_{unq_name}"
        )

    @staticmethod
    def get_collection_name(unq_name: str = None):
        return "Cylinder_Human" if unq_name is None else f"Cylinder_Human_{unq_name}"


class StickFigure:
    @staticmethod
    def visualize(
        joint_locations: np.ndarray,
        bone_list=BONES_24_INDS,
        unq_name=None,
        color=None,
        skeleton_size_map: SkeletonSizeMap = None,
    ):
        """
        joint locations: [seq_len, n_joints, 3]
        name: use in case of multiple stick figures visualizations
        """
        collection_name = Names.get_collection_name(unq_name)
        CollectionManager.create_collection(collection_name)
        cyl_bone_dict = StickFigure.create_cylinder_bones(
            collection_name, bone_list, unq_name, skeleton_size_map
        )
        StickFigure.create_joints(
            collection_name, bone_list, unq_name, skeleton_size_map
        )
        n_frames = joint_locations.shape[0]
        for frame in tqdm(
            range(n_frames),
            desc=f"Stick figure sequence{f' {unq_name}' if unq_name else ''}",
        ):
            pose = joint_locations[frame]
            StickFigure.apply_pose(pose, frame, cyl_bone_dict, bone_list)

        if color is not None:
            material_name = f"Mat_{collection_name}"
            material = bpy.data.materials.get(material_name)
            if material is None:
                material = bpy.data.materials.new(name=material_name)
            material.diffuse_color = color if len(color) == 4 else color + (1.0,)
            for obj in bpy.data.collections[collection_name].objects:
                if obj.type == "MESH":
                    obj.data.materials.clear()
                    obj.data.materials.append(material)

    @staticmethod
    def create_cylinder_bones(
        collection_name,
        bone_list,
        unq_name,
        skeleton_size_mapper: SkeletonSizeMap = None,
    ) -> Dict[Tuple[int, int], CylinderBone]:
        cyl_bones = {}
        for head, tail in bone_list:
            radius = (
                0.03
                if skeleton_size_mapper is None
                else skeleton_size_mapper.get_bone_size(head, tail)
            )
            cyl_bone = CylinderBone(head, tail, unq_name, radius=radius)
            cyl_bones[(head, tail)] = cyl_bone
            CollectionManager.add_object_to_collection(
                cyl_bone.bone_name, collection_name
            )
        return cyl_bones

    @staticmethod
    def create_joints(
        collection_name,
        bone_list: List[Tuple[int, int]],
        unq_name: str,
        skeleton_size_mapper: SkeletonSizeMap = None,
    ):
        joints = set()
        for bone in bone_list:
            for j in bone:
                if j not in joints:
                    radius = (
                        0.04
                        if skeleton_size_mapper is None
                        else skeleton_size_mapper.get_joint_size(j)
                    )
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=radius,
                        calc_uvs=False,
                        enter_editmode=False,
                        align="WORLD",
                        scale=(1, 1, 1),
                    )
                    joint_name = Names.get_joint_name(j, unq_name)
                    bpy.context.object.name = joint_name
                    CollectionManager.add_object_to_collection(
                        joint_name, collection_name
                    )
                    joints.add(j)

    @staticmethod
    def apply_pose(pose, frame: int, cyl_bone_dict: Dict[str, CylinderBone], bone_list):
        """
        pose: (n_joints, 3)
        """
        for head_ind, tail_ind in bone_list:
            head_location = pose[head_ind]
            tail_location = pose[tail_ind]
            cyl_bone = cyl_bone_dict[(head_ind, tail_ind)]
            cyl_bone.set_head_and_tail_locations_per_frame(
                head_location, tail_location, frame=frame
            )


VERTICES_TO_MODEL = {
    6890: "smpl",
    10475: "smplx",
}


def visualize_motions(
    motion_lst,
    save_path="./stick_figures.blend",
    targets=None,
    fps=20,
    bone_list=None,
    zup=True,
    mat_lst=None,
    print_download_and_run=True,
    verts_anim_lst=None,
    colors=None,
    skeleton_size_map: Union[SkeletonSizeMap, str] = "default",
    unq_name="",
    text_metadata=None,
    reset_blender=True,
):

    n_joints = motion_lst[0].shape[1]
    if bone_list is None:
        if n_joints == 24:
            bone_list = BONES_24_INDS
        elif n_joints == 52:
            bone_list = BONE_52_52_INDS
        else:
            raise ValueError()

    if skeleton_size_map == "default":
        if n_joints <= 24:
            skeleton_size_map = DefaultSkeletonSizeMap(24)
        else:
            skeleton_size_map = DefaultSkeletonSizeMap(22)

    #################
    # Pre-process inputs
    #################
    motion_lst = [
        m.detach().clone().cpu().numpy() if not isinstance(m, np.ndarray) else m
        for m in motion_lst
    ]
    motion_lst = motion_lst if zup else [yup2zup(motion) for motion in motion_lst]

    if verts_anim_lst is not None:
        verts_anim_lst = [
            v.detach().clone().cpu().numpy() if not isinstance(v, np.ndarray) else v
            for v in verts_anim_lst
        ]
        verts_anim_lst = (
            verts_anim_lst if zup else [yup2zup(anim) for anim in verts_anim_lst]
        )

    if mat_lst is not None:
        mat_lst = mat_lst if zup else [yup2zup(mat) for mat in mat_lst]

    if reset_blender:
        delete_all_data()

    duration = max([m.shape[0] for m in motion_lst])
    set_duration(duration)

    if colors in [None, "pairs"]:
        if colors is None:
            cmap = plt.get_cmap("tab10")
        elif colors == "pairs":
            cmap = plt.get_cmap("tab20")
        colors = cmap.colors
        if len(colors) < len(motion_lst):
            colors = colors * math.ceil(len(motion_lst) / len(colors))

            # colors = []
            # for i in range(len(motion_lst)):
            #     c = cmap.colors[i]
            #     c1 = np.concatenate([np.clip(np.array(c) * 0.7, 0, 1), [1]])
            #     c2 = np.concatenate([np.clip(np.array(c) * 1.3, 0, 1), [1]])
            #     colors += [c1, c2]

    #################
    # Visualize meshes
    #################
    if verts_anim_lst:
        n_vers = verts_anim_lst[0][0].shape[0]
        model_name = VERTICES_TO_MODEL[n_vers]
        sbj_mesh_faces, _ = get_smpl_template(model_name)
        for i, verts_anim in enumerate(verts_anim_lst):
            animate_mesh(f"{unq_name}SMPLX_sample{i}", sbj_mesh_faces, verts_anim)

    #################
    # Visualize stick figures
    #################
    for i, motion in enumerate(motion_lst):
        StickFigure.visualize(
            motion,
            unq_name=f"{unq_name}sample{i}",
            color=colors[i],
            bone_list=bone_list,
            skeleton_size_map=skeleton_size_map,
        )

    #################
    # Visualize static axes
    #################
    if mat_lst is not None:
        for i, mat in enumerate(mat_lst):
            create_xyz_axes(transform_matrix=mat, unq_name=f"{unq_name}AXES{i}")

    #################
    # Visualize static targets
    #################
    if targets is not None:
        targets = yup2zup(targets)
        for i in range(len(targets)):
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=targets[i].squeeze(),
                radius=0.04,
                calc_uvs=False,
                enter_editmode=False,
                align="WORLD",
                scale=(1, 1, 1),
            )
            # sphere = bpy.context.object
            # sphere.name = f"target_{i}"
            # sphere.location = targets[i]
            # sphere.keyframe_insert(data_path="location", frame=0)

    bpy.context.scene.render.fps = fps

    if text_metadata is not None:
        text_block = bpy.data.texts.new("Metadata")
        text_block.write(text_metadata)

    if save_path is not None:
        save_blender_file(save_path)
        if print_download_and_run:
            blend_scp_and_run(save_path)


def visualize_vertices(
    vertiecs: np.ndarray, save_path=".vertices_animation.blend", fps=30, max_points=1000
):
    """
    vertices.shape = (seq_len, N_verts, 3)
    """
    vertiecs = (
        vertiecs.detach().clone().cpu().numpy()
        if not isinstance(vertiecs, np.ndarray)
        else vertiecs
    )
    vertiecs = yup2zup(vertiecs)
    if reset_blender:
        delete_all_data()
    duration = vertiecs.shape[0]
    set_duration(duration)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(1)]
    color = colors[0]
    create_spheres_animation(vertiecs, max_points=max_points)
    bpy.context.scene.render.fps = fps
    if save_path is not None:
        save_blender_file(save_path)


def main():
    save_path = "./hsi_output.blend"
    video_path = None

    with open("output.pkl", "rb") as f:
        d = pickle.load(f)

    # motions assumed to be [seq_len, n_joints, 3]
    motion_init = yup2zup(d["initial_motion"])
    motion_opti = yup2zup(d["optimized_motion"])
    motion_3 = yup2zup(np.load("/home/dcor/roeyron/trumans_utils/sample_joints.npy"))

    delete_all_data()
    add_camera()
    add_chessboard_floor()
    add_lights()

    duration = max(motion_init.shape[0], motion_opti.shape[0], motion_3.shape[0])
    set_duration(duration)

    StickFigure.visualize(motion_init, unq_name="original")
    StickFigure.visualize(motion_opti, unq_name="optimized")
    StickFigure.visualize(motion_3, unq_name="motion3")

    if save_path is not None:
        save_blender_file(save_path)
    if video_path is not None:
        render(video_path)


if __name__ == "__main__":
    main()
