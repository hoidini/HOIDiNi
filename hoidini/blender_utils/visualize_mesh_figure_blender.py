import bpy
import torch
from tqdm import tqdm
import numpy as np
import pickle
from hoidini.amasstools.geometry import axis_angle_to_quaternion
import hoidini.smplx as smplx
from hoidini.datasets.smpldata import get_smpl_model_path


def get_smpl_template(model_name, gender="neutral", smpl_models_path=None):
    smpl_models_path = get_smpl_model_path(model_name, gender, smpl_models_path)
    with open(smpl_models_path, "rb") as f:
        d = pickle.load(f, encoding="latin")
    sbj_mesh_faces = d["f"]
    sbj_mesh_vertices = d["v_template"]
    return sbj_mesh_faces, sbj_mesh_vertices


def create_new_blender_object_with_mesh(name, initial_verts, faces, color=None):
    mesh = bpy.data.meshes.new(name=f"Mesh_{name}")
    mesh.from_pydata(initial_verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name=f"Mesh_{name}", object_data=mesh)
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    if color is not None:
        material_name = f"Mat_{name}"
        material = bpy.data.materials.get(material_name)
        if material is None:
            material = bpy.data.materials.new(name=material_name)
        material.diffuse_color = color if len(color) == 4 else color + (1.0,)

        obj.data.materials.clear()
        obj.data.materials.append(material)
    return obj


def animate_mesh(name, mesh_faces, verts_anim, color=None):
    if isinstance(mesh_faces, torch.Tensor):
        mesh_faces = mesh_faces.cpu().numpy()
    if isinstance(verts_anim, torch.Tensor):
        verts_anim = verts_anim.cpu().numpy()
    sbj_blend_obj = create_new_blender_object_with_mesh(
        name, verts_anim[0], mesh_faces, color=color
    )
    sbj_blend_verts = sbj_blend_obj.data.vertices
    for frame in tqdm(range(len(verts_anim)), desc=f"animate mesh {name}"):
        for v_i, v in enumerate(sbj_blend_verts):
            v.co = verts_anim[frame][v_i]
            v.keyframe_insert("co", frame=frame)


def animate_rigid_mesh(name, mesh_faces, v_template, poses, transl, color=None):
    """
    poses is in axis-angle format
    """
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses)
    else:
        poses = poses.cpu()
    quats_wxyz = axis_angle_to_quaternion(poses)
    quats_wxyz = quats_wxyz.numpy()
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    mesh_faces = (
        mesh_faces.cpu().numpy() if isinstance(mesh_faces, torch.Tensor) else mesh_faces
    )
    v_template = (
        v_template.cpu().numpy() if isinstance(v_template, torch.Tensor) else v_template
    )
    transl = transl.cpu().numpy() if isinstance(transl, torch.Tensor) else transl
    blend_obj = create_new_blender_object_with_mesh(
        name, v_template, mesh_faces, color=color
    )
    for frame in tqdm(range(len(transl)), desc=f"animate rigid mesh {name}"):
        blend_obj.location = transl[frame]
        blend_obj.rotation_quaternion = quats_xyzw[frame]
        blend_obj.keyframe_insert("location", frame=frame)
        blend_obj.keyframe_insert("rotation_quaternion", frame=frame)


def main():
    smplx_output = smplx.utils.SMPLOutput()
    sbj_mesh_faces, _ = get_smpl_template()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sbj_verts_anim = smplx_output.vertices.detach().cpu().numpy()
    animate_mesh("SMPLX", sbj_mesh_faces, sbj_verts_anim)
