import bpy
import torch
import bmesh


def add_table_from_4_points(points, name="TablePlane"):
    """
    Adds a quad surface mesh to the scene, formed by four given 3D points.

    Args:
        points (list of tuple): A list of 4 tuples, each representing (x, y, z) coordinates.
        name (str): Name for the created mesh and object.
    """
    assert len(points) == 4, "Expected exactly 4 points to form a quad."
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Build mesh using bmesh
    bm = bmesh.new()
    verts = [bm.verts.new(co) for co in points]
    bm.faces.new(verts)
    bm.to_mesh(mesh)
    bm.free()


def main():
    table_points = [
        [-0.3222, -0.9099, 0.7420],
        [0.1253, -0.9193, 0.7352],
        [0.1353, -0.3779, 0.7300],
        [-0.3135, -0.3743, 0.7349],
    ]
    add_table_from_4_points(table_points)


if __name__ == "__main__":
    main()
