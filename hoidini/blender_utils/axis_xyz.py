import bpy
import math
from mathutils import Vector, Matrix, Euler


def matrix_to_blender(matrix):
    """Convert numpy or torch matrix to Blender matrix"""
    if matrix is None:
        return Matrix.Identity(4)

    # Check if it's a numpy array
    if str(type(matrix).__module__) == "numpy":
        return Matrix(
            [
                [
                    float(matrix[0, 0]),
                    float(matrix[0, 1]),
                    float(matrix[0, 2]),
                    float(matrix[0, 3]),
                ],
                [
                    float(matrix[1, 0]),
                    float(matrix[1, 1]),
                    float(matrix[1, 2]),
                    float(matrix[1, 3]),
                ],
                [
                    float(matrix[2, 0]),
                    float(matrix[2, 1]),
                    float(matrix[2, 2]),
                    float(matrix[2, 3]),
                ],
                [
                    float(matrix[3, 0]),
                    float(matrix[3, 1]),
                    float(matrix[3, 2]),
                    float(matrix[3, 3]),
                ],
            ]
        )

    # Check if it's a torch tensor
    elif str(type(matrix).__module__) == "torch":
        matrix_np = matrix.detach().cpu().numpy()
        return Matrix(
            [
                [
                    float(matrix_np[0, 0]),
                    float(matrix_np[0, 1]),
                    float(matrix_np[0, 2]),
                    float(matrix_np[0, 3]),
                ],
                [
                    float(matrix_np[1, 0]),
                    float(matrix_np[1, 1]),
                    float(matrix_np[1, 2]),
                    float(matrix_np[1, 3]),
                ],
                [
                    float(matrix_np[2, 0]),
                    float(matrix_np[2, 1]),
                    float(matrix_np[2, 2]),
                    float(matrix_np[2, 3]),
                ],
                [
                    float(matrix_np[3, 0]),
                    float(matrix_np[3, 1]),
                    float(matrix_np[3, 2]),
                    float(matrix_np[3, 3]),
                ],
            ]
        )

    # If it's already a Blender matrix, return as is
    elif isinstance(matrix, Matrix):
        return matrix

    # If it's a list or tuple, convert to Matrix
    elif isinstance(matrix, (list, tuple)):
        return Matrix(matrix)

    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")


def create_cylinder(radius, height, location, rotation, collection):
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=height, location=location)
    cylinder = bpy.context.active_object
    cylinder.rotation_euler = rotation

    # Move to correct collection
    for col in cylinder.users_collection:
        col.objects.unlink(cylinder)
    collection.objects.link(cylinder)

    return cylinder


def create_xyz_axes(
    size=1, radius=0.007, material=None, transform_matrix=None, unq_name: str = ""
):
    # Convert input matrix to Blender matrix
    transform_matrix = matrix_to_blender(transform_matrix)

    # Find unique collection name
    i = 0
    while True:
        collection_name = f"XYZ_Axes{unq_name}{i}"
        if collection_name not in bpy.data.collections:
            break
        i += 1

    # Create new collection
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    # Extract translation and rotation from matrix
    translation = transform_matrix.to_translation()
    rotation_matrix = transform_matrix.to_3x3()

    # Create cylinders for each axis
    axes = []

    # Helper function to transform positions and rotations
    def transform_cylinder_params(local_pos, local_rot):
        # Transform position
        world_pos = transform_matrix @ Vector((*local_pos, 1))

        # Transform rotation
        local_euler = Euler(local_rot)
        local_matrix = local_euler.to_matrix().to_4x4()
        world_matrix = transform_matrix @ local_matrix
        world_euler = world_matrix.to_euler()

        return world_pos[:3], world_euler

    def create_sphere(rad, loc, name):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=rad, location=loc)
        sphere = bpy.context.active_object
        for ccol in sphere.users_collection:
            ccol.objects.unlink(sphere)
        collection.objects.link(sphere)
        sphere.name = f"{name}_Sphere{i}"
        return sphere

    # X axis (red)
    pos, rot = transform_cylinder_params((size / 2, 0, 0), (0, math.pi / 2, 0))
    x_cyl = create_cylinder(
        radius=radius, height=size, location=pos, rotation=rot, collection=collection
    )
    x_cyl.name = f"X_Axis{i}"
    axes.append((x_cyl.name, (1, 0, 0, 1)))  # Red

    # Y axis (green)
    pos, rot = transform_cylinder_params((0, size / 2, 0), (math.pi / 2, 0, 0))
    y_cyl = create_cylinder(
        radius=radius, height=size, location=pos, rotation=rot, collection=collection
    )
    y_cyl.name = f"Y_Axis{i}"
    axes.append((y_cyl.name, (0, 1, 0, 1)))  # Green

    # Z axis (blue)
    pos, rot = transform_cylinder_params((0, 0, size / 2), (0, 0, 0))
    z_cyl = create_cylinder(
        radius=radius, height=size, location=pos, rotation=rot, collection=collection
    )
    z_cyl.name = f"Z_Axis{i}"
    axes.append((z_cyl.name, (0, 0, 1, 1)))  # Blue

    # Add one sphere at the connection point (origin)
    center_loc, _ = transform_cylinder_params((0, 0, 0), (0, 0, 0))
    center_sphere = create_sphere(radius * 2, center_loc, "XYZ_Center")
    axes.append((center_sphere.name, (1, 1, 1, 1)))  # White

    # Handle materials
    if material is not None:
        # Use provided material for all axes
        for obj_name, _ in axes:
            obj = collection.objects[obj_name]
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material)
            else:
                obj.data.materials[0] = material
    else:
        # Create and assign default colored materials
        for obj_name, color in axes:
            material_name = f"Mat_{obj_name}"
            material = bpy.data.materials.get(material_name)
            if material is None:
                material = bpy.data.materials.new(name=material_name)
            material.use_nodes = False  # Use simple material
            material.diffuse_color = color

            obj = collection.objects[obj_name]
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material)
            else:
                obj.data.materials[0] = material

    return collection


if __name__ == "__main__":
    # Example usage:
    # Create default axes at origin
    axes_collection1 = create_xyz_axes(size=1.0, radius=0.07)

# Using numpy matrix
"""
import numpy as np
transform_np = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])
axes_collection2 = create_xyz_axes(size=1.0, radius=0.07, transform_matrix=transform_np)
"""

# Using PyTorch matrix
"""
import torch
transform_torch = torch.tensor([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
], dtype=torch.float32)
axes_collection3 = create_xyz_axes(size=1.0, radius=0.07, transform_matrix=transform_torch)
"""
