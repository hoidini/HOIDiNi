from multiprocessing import current_process
import os
from typing import List, Optional, Union

if current_process().name == "MainProcess":
    import bpy


class ObjectHook:
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


def hide_collection(collection_name: Union[str, List[str]], hide: bool = True):
    def find_layer_collection(layer_collection, target_collection):
        if layer_collection.collection == target_collection:
            return layer_collection
        for child in layer_collection.children:
            found = find_layer_collection(child, target_collection)
            if found:
                return found
        return None

    if isinstance(collection_name, str):
        collection_names = [collection_name]
    else:
        collection_names = collection_name

    for name in collection_names:
        collection = bpy.data.collections.get(name)
        if collection is None:
            print(f"Collection '{name}' not found.")
            continue

        for view_layer in bpy.context.scene.view_layers:
            layer_collection = find_layer_collection(
                view_layer.layer_collection, collection
            )
            if layer_collection:
                layer_collection.exclude = (
                    hide  # This controls the visibility toggle in UI
                )


class CollectionManager:
    @staticmethod
    def create_collection(collection_name):
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection)

    @staticmethod
    def add_object_to_collection(object_name, collection_name):
        object_to_add = bpy.data.objects.get(object_name)
        for other_collection in object_to_add.users_collection:
            other_collection.objects.unlink(object_to_add)
        collection = bpy.data.collections[collection_name]
        collection.objects.link(object_to_add)

    @staticmethod
    def add_collection_to_collection(collection_name, target_collection_name):
        collection = bpy.data.collections.get(collection_name)
        target_collection = bpy.data.collections.get(target_collection_name)

        if collection and target_collection:
            # Unlink from main scene collection if present
            for child in bpy.context.scene.collection.children:
                if child.name == collection_name:
                    bpy.context.scene.collection.children.unlink(collection)
                    break

            # Find and unlink from other parents
            for parent in bpy.data.collections:
                for child in parent.children:
                    if child.name == collection_name:
                        parent.children.unlink(collection)
                        break

            # Link to new parent
            target_collection.children.link(collection)

    @staticmethod
    def organize_collections(substr_map: dict | str | List[str]):
        """
        substr_map: {"substr_of_object_or_collection": "target_collection"}
        Will move the objects/collections that match the substr to the target collection and will create the target collection if it doesn't exist
        Ignore objects which are not direction children of the main collection
        """
        if isinstance(substr_map, str):
            substr_map = {substr_map: substr_map}
        if isinstance(substr_map, list):
            substr_map = {e: e for e in substr_map}

        # Get all direct children of the main collection
        main_collection = bpy.context.scene.collection
        direct_children = [obj for obj in main_collection.objects]
        direct_collections = [coll for coll in main_collection.children]

        # Process each mapping
        for substr, target_name in substr_map.items():
            # Create target collection if it doesn't exist
            if target_name is None:
                continue
            target_collection = bpy.data.collections.get(target_name)
            if not target_collection:
                target_collection = bpy.data.collections.new(target_name)
                main_collection.children.link(target_collection)

            # Move matching objects
            for obj in direct_children:
                if substr in obj.name:
                    CollectionManager.add_object_to_collection(obj.name, target_name)

            # Move matching collections
            for coll in direct_collections:
                if substr in coll.name:
                    CollectionManager.add_collection_to_collection(
                        coll.name, target_name
                    )


def delete_all_data():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    for coll in bpy.data.collections:
        if coll != bpy.data.collections["Collection"]:
            bpy.data.collections.remove(coll)

    # Delete all armatures
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)


def save_blender_file(path="/tmp/debug.blend"):
    print(f"{10 * '*'} saved {path}")
    bpy.ops.wm.save_as_mainfile(filepath=path)


def reset_blender():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def blend_scp_and_run(save_path):
    src = os.path.abspath(save_path)
    dst = f"~/Downloads/{os.path.basename(save_path)}"
    user = None
    try:
        user = os.getlogin()
    except OSError:
        pass
    if user is None:
        user = os.environ.get("SLURM_JOB_USER", None)
    if user is None:
        user = "roeyron"
    print(f"\nscp {user}@c-006.cs.tau.ac.il:{src} {dst}\nblender {dst}\n")


def set_fps(fps: int):
    bpy.context.scene.render.fps = fps
    bpy.context.scene.render.fps_base = 1.0


def set_frame_end(
    *, start_frame: Optional[int] = None, end_frame: Optional[int] = None
):
    if start_frame is not None:
        bpy.context.scene.frame_start = start_frame
    if end_frame is not None:
        bpy.context.scene.frame_end = end_frame


"""
/Applications/Blender.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install 
"""
