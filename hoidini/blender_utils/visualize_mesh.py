import glob
import numpy as np
import bpy
from pytorch3d.structures import Meshes
import trimesh
import os

from hoidini.blender_utils.general_blender_utils import blend_scp_and_run
from hoidini.datasets.grab.grab_utils import simplify_trimesh
from hoidini.general_utils import TMP_DIR
from hoidini.resource_paths import GRAB_DATA_PATH


def create_mesh(vertices, faces, mesh_name="MyMesh", obj_name="MyMesh"):
    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return obj


def visualize_mesh(
    verts: np.ndarray = None,
    faces: np.ndarray = None,
    mesh: Meshes = None,
    save_blend_fname=None,
    unq_name="",
    reset_blender=True,
):
    if reset_blender:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    # Example usage:
    if mesh is not None:
        verts = mesh.verts_list()[0].detach().cpu().numpy()
        faces = mesh.faces_list()[0].detach().cpu().numpy()
    create_mesh(verts, faces, mesh_name=unq_name, obj_name=unq_name)
    save_blend_fname = save_blend_fname or "/home/dcor/roeyron/tmp/mymesh.blend"
    bpy.ops.wm.save_as_mainfile(filepath=save_blend_fname)
    print(
        f"scp roeyron@c-005.cs.tau.ac.il:{os.path.abspath(save_blend_fname)} ~/Downloads/{os.path.basename(save_blend_fname)}\nblender ~/Downloads/{os.path.basename(save_blend_fname)}"
    )


def show_static_mesh():
    obj_mesh_paths = [
        os.path.join(GRAB_DATA_PATH, p)
        for p in sorted(glob.glob("**/*.ply", root_dir=GRAB_DATA_PATH))
    ]
    # obj_mesh_path = obj_mesh_paths[0]
    n_simplify_object = 1000

    for i, obj_mesh_path in enumerate(obj_mesh_paths):
        print(obj_mesh_path)
        save_path = os.path.join(
            TMP_DIR, f"{os.path.basename(obj_mesh_path).replace('.ply', '.blend')}"
        )
        obj_mesh = trimesh.load(obj_mesh_path)
        obj_mesh = simplify_trimesh(obj_mesh, tgt_faces=n_simplify_object)
        unq_name = obj_mesh_path.split("/")[-1].replace(".ply", "")
        visualize_mesh(
            np.array(obj_mesh.vertices) + np.array([0.5 * i, 0, 0]),
            np.array(obj_mesh.faces),
            unq_name=unq_name,
            reset_blender=False,
        )
    bpy.ops.wm.save_mainfile(filepath=save_path)
    blend_scp_and_run(save_path)


if __name__ == "__main__":
    show_static_mesh()
