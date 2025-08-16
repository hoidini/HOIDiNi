import bpy
from mathutils import Vector, Euler


def add_light():
    import bpy

    # Delete default lamp if it exists
    if "Light" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)

    # Add a new light object
    light_data = bpy.data.lights.new(
        name="MyKeyLight", type="SUN"
    )  # Use 'POINT', 'SUN', 'SPOT', 'AREA' etc.
    light_object = bpy.data.objects.new(name="MyKeyLight", object_data=light_data)

    # Set the light location and energy
    light_object.location = (5, -5, 5)  # Change location as needed
    light_data.energy = 5.0  # Adjust brightness

    # Link the light object to the current collection
    bpy.context.collection.objects.link(light_object)

    # Optional: Add a second fill light
    fill_data = bpy.data.lights.new(name="FillLight", type="AREA")
    fill_object = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    fill_object.location = (-5, 5, 3)
    fill_data.energy = 2.0
    bpy.context.collection.objects.link(fill_object)

    # Optional: Set scene to use 'Cycles' renderer for better lighting
    bpy.context.scene.render.engine = "CYCLES"


def freeze():
    import bpy

    frame = bpy.context.scene.frame_current
    for obj in bpy.context.selected_objects:
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)


def add_point_on_vertices(verts, faces):
    import bpy
    import bmesh

    # Get the active object
    obj = bpy.context.object

    bpy.ops.object.mode_set(mode="EDIT")

    # Get the bmesh representation
    # mesh = bmesh.from_edit_mesh(obj.data)

    # Get the selected vertex indices
    # selected_verts = [v.index for v in mesh.verts if v.select]

    vert_ids = [
        48,
        61,
        73,
        95,
        98,
        110,
        114,
        115,
        125,
        130,
        147,
        149,
        171,
        341,
        350,
        358,
        379,
        453,
        462,
        470,
        489,
        564,
        573,
        582,
        607,
        681,
        690,
        754,
        763,
        770,
    ]

    for v_ind in vert_ids:
        # loc = mesh.verts[v_ind].location

        vertex = obj.data.vertices[v_ind]

        location = obj.matrix_world @ vertex.co
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.005, location=location)


def add_chessboard_floor(board_size=10, tile_size=1.0):
    """
    Adds a chessboard floor to the scene.

    Parameters:
    - board_size (int): Number of tiles per side (e.g., 8 for 8x8).
    - tile_size (float): Size of each tile (square) in meters.
    """
    total_size = board_size * tile_size

    # Create a new plane with the correct size (no scaling needed)
    bpy.ops.mesh.primitive_plane_add(
        size=total_size, enter_editmode=False, location=(0, 0, 0)
    )
    plane = bpy.context.object

    # Create a new material with checkerboard texture
    material = bpy.data.materials.new(name="Checkered_Material")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    # Create and configure nodes
    checker_tex = nodes.new("ShaderNodeTexChecker")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")

    # Make checker grid match the number of tiles
    mapping.inputs["Scale"].default_value = (board_size, board_size, 1)

    # Connect nodes
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker_tex.inputs["Vector"])
    links.new(checker_tex.outputs["Color"], bsdf.inputs["Base Color"])

    # Assign material
    plane.data.materials.append(material)


def add_camera():
    import bpy

    bpy.ops.object.camera_add(
        location=(2.7143783569335938, -3.934565782546997, 2.3345868587493896),
        rotation=Euler(
            (64.09072875976562, -1.768244288768983e-07, 19.448474884033203), "XYZ"
        ),
    )
    camera = bpy.context.object
    camera.name = "MainCamera"
    return camera


def generate_light_creation_code():
    import bpy

    code_lines = []
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            light = obj.data
            code_lines.append(f"# Light: {obj.name}")
            code_lines.append(
                f"light_data = bpy.data.lights.new(name='{light.name}', type='{light.type}')"
            )
            code_lines.append(f"light_data.energy = {light.energy}")
            code_lines.append(
                f"light_data.color = ({light.color[0]}, {light.color[1]}, {light.color[2]})"
            )
            if light.type == "AREA":
                code_lines.append(f"light_data.size = {light.size}")
            if light.type == "SPOT":
                code_lines.append(f"light_data.spot_size = {light.spot_size}")
                code_lines.append(f"light_data.spot_blend = {light.spot_blend}")
            code_lines.append(
                f"light_object = bpy.data.objects.new(name='{obj.name}', object_data=light_data)"
            )
            code_lines.append(
                f"light_object.location = ({obj.location[0]}, {obj.location[1]}, {obj.location[2]})"
            )
            code_lines.append(
                f"light_object.rotation_euler = ({obj.rotation_euler[0]}, {obj.rotation_euler[1]}, {obj.rotation_euler[2]})"
            )
            code_lines.append(f"bpy.context.collection.objects.link(light_object)\n")
    return "\n".join(code_lines)


def generate_light_creation_function(function_name="create_lights"):
    import bpy

    code_lines = []
    code_lines.append(f"def {function_name}():")
    code_lines.append(f"    import bpy\n")

    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            light = obj.data
            code_lines.append(f"    # Light: {obj.name}")
            code_lines.append(
                f"    light_data = bpy.data.lights.new(name='{light.name}', type='{light.type}')"
            )
            code_lines.append(f"    light_data.energy = {light.energy}")
            code_lines.append(
                f"    light_data.color = ({light.color[0]:.6f}, {light.color[1]:.6f}, {light.color[2]:.6f})"
            )

            if light.type == "AREA":
                code_lines.append(f"    light_data.size = {light.size}")
            if light.type == "SPOT":
                code_lines.append(f"    light_data.spot_size = {light.spot_size}")
                code_lines.append(f"    light_data.spot_blend = {light.spot_blend}")

            code_lines.append(
                f"    light_object = bpy.data.objects.new(name='{obj.name}', object_data=light_data)"
            )
            code_lines.append(
                f"    light_object.location = ({obj.location[0]:.6f}, {obj.location[1]:.6f}, {obj.location[2]:.6f})"
            )
            code_lines.append(
                f"    light_object.rotation_euler = ({obj.rotation_euler[0]:.6f}, {obj.rotation_euler[1]:.6f}, {obj.rotation_euler[2]:.6f})"
            )
            code_lines.append(
                f"    bpy.context.collection.objects.link(light_object)\n"
            )

    return "\n".join(code_lines)


def create_lights():
    import bpy

    # Light: Area
    light_data = bpy.data.lights.new(name="Area", type="AREA")
    light_data.energy = 500.0
    light_data.color = (1.000000, 1.000000, 1.000000)
    light_data.size = 1.0
    light_object = bpy.data.objects.new(name="Area", object_data=light_data)
    light_object.location = (3.367397, -6.767385, 2.841775)
    light_object.rotation_euler = (1.327957, 0.327856, 0.062960)
    bpy.context.collection.objects.link(light_object)

    # Light: Area.001
    light_data = bpy.data.lights.new(name="Area.001", type="AREA")
    light_data.energy = 348.79998779296875
    light_data.color = (1.000000, 1.000000, 1.000000)
    light_data.size = 1.0
    light_object = bpy.data.objects.new(name="Area.001", object_data=light_data)
    light_object.location = (-1.204519, 2.415360, 1.824571)
    light_object.rotation_euler = (-1.144594, 0.064281, 0.375649)
    bpy.context.collection.objects.link(light_object)

    # Light: Area.002
    light_data = bpy.data.lights.new(name="Area.002", type="AREA")
    light_data.energy = 36.39999771118164
    light_data.color = (1.000000, 1.000000, 1.000000)
    light_data.size = 1.0
    light_object = bpy.data.objects.new(name="Area.002", object_data=light_data)
    light_object.location = (-2.070962, -3.843272, 1.700447)
    light_object.rotation_euler = (1.531489, -0.000000, -0.711098)
    bpy.context.collection.objects.link(light_object)

    # Light: Point
    light_data = bpy.data.lights.new(name="Point", type="POINT")
    light_data.energy = 21.099998474121094
    light_data.color = (1.000000, 1.000000, 1.000000)
    light_object = bpy.data.objects.new(name="Point", object_data=light_data)
    light_object.location = (1.787582, -2.608638, 1.498101)
    light_object.rotation_euler = (0.000000, 0.000000, 0.000000)
    bpy.context.collection.objects.link(light_object)

    # Light: Sun
    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light_data.energy = 1.0
    light_data.color = (1.000000, 1.000000, 1.000000)
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    light_object.location = (1.535658, -1.185509, 2.060576)
    light_object.rotation_euler = (1.154156, -0.000000, 0.878172)
    bpy.context.collection.objects.link(light_object)


def stick_contact_pairs_to_mesh():
    import bpy
    import bmesh
    import mathutils
    from mathutils import Vector
    from bpy import context

    # CONFIG
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    target_object_name = (
        "Mesh_Static_ObjContactPairs0"  # <-- Replace with your mesh name
    )
    point_collection_name = (
        "ContactsLocalContactPairs0"  # <-- Replace with your points collection
    )

    # Get target mesh and its vertex world coordinates
    target_obj = bpy.data.objects[target_object_name]
    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bm.verts.ensure_lookup_table()

    # Cache vertex world positions
    target_verts_world = [target_obj.matrix_world @ v.co for v in bm.verts]

    # Get animated point objects
    point_objs = list(bpy.data.collections[point_collection_name].objects)

    # Iterate over frames
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)

        for point in point_objs:
            point_loc = point.matrix_world.translation

            # Find nearest vertex
            closest_vert = min(
                target_verts_world, key=lambda v: (v - point_loc).length_squared
            )

            # Move point to nearest vertex
            point.location = closest_vert

            # Insert keyframe
            point.keyframe_insert(data_path="location", frame=frame)

    bm.free()


# add_camera()
# add_light()
# freeze()
# add_chessboard_floor(board_size=20, tile_size=3.0)
# print(generate_light_creation_function())
create_lights()
