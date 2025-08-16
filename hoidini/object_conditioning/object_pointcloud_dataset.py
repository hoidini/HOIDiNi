import os
import shutil
from typing import List, Tuple
import torch
import numpy as np
import open3d as o3d
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
import torch_geometric.transforms as T
from tqdm import tqdm

from hoidini.general_utils import TMP_DIR
from hoidini.resource_paths import GRAB_DATA_PATH


def read_ply(filename) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float)
    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long)
    normals = torch.tensor(np.asarray(mesh.vertex_normals), dtype=torch.float)
    return vertices, faces, normals


class RandomApply:
    """Apply an inner transform with probability p."""

    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, data):
        if torch.rand(1).item() < self.p:
            data = self.transform(data)
        return data


# class InTheWildObjPcdDataset(InMemoryDataset):
#     """
#     Return a PyGData object for each object in the dataset.
#     """
#     def __init__(self, root, file_list, transform=None, pre_transform=None, pre_filter=None, force_reload: bool = True):
#         self.file_list = file_list
#         self.force_reload = force_reload


class GrabObjPcdDataset(InMemoryDataset):
    """
    Return a PyGData object for each object in the dataset.
    """

    def __init__(
        self,
        root,
        file_list,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = True,
    ):
        self.file_list = file_list
        self.force_reload = force_reload
        super(GrabObjPcdDataset, self).__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.processed_paths[0])

        self.name2idx = {
            os.path.basename(file).split(".")[0]: i
            for i, file in enumerate(self.file_list)
        }

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list = []
        for file in tqdm(self.file_list, desc="Processing point clouds"):
            vertices, faces, normals = read_ply(file)
            data = Data(
                pos=vertices, face=faces.t(), normal=normals
            )  # vertices are (V, 3), faces are (3, F), normals are (V, 3)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self):
        return f"{self.__class__.__name__}(num_samples={len(self)})"


def get_in_the_wild_object_point_cloud(obj_path, n_points: int = 128):
    """
    Loads an in-the-wild object mesh and returns a single processed point cloud datapoint (PyG Data object).
    Args:
        obj_path: Path to the mesh file (e.g., .ply, .obj).
        n_points: Number of points to sample from the mesh surface.
    Returns:
        torch_geometric.data.Data: A single point cloud datapoint.
    """
    vertices, faces, normals = read_ply(obj_path)
    data = Data(pos=vertices, face=faces.t(), normal=normals)

    # Pre-transform: center
    data = T.Center()(data)

    # Transform: sample points (with normals)
    data = T.SamplePoints(n_points, include_normals=True)(data)

    return data


def get_grab_point_cloud_dataset(
    grab_dataset_path: str,
    use_cache: bool = False,
    n_points: int = 128,
    augment_rot_z: bool = False,
    augment_jitter: bool = False,
) -> GrabObjPcdDataset:
    ply_files_dir = os.path.join(grab_dataset_path, "contact_meshes")
    cache_dir = os.path.join(TMP_DIR, "grab_objects_point_cloud_dataset")
    if not use_cache and os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)  # for debugging
        except Exception as e:
            print(f"Error removing cache directory: {e}")
    file_list = os.listdir(ply_files_dir)
    file_list = [os.path.join(ply_files_dir, f) for f in file_list]

    pre_transform_lst = [T.Center()]
    transform_lst = [T.SamplePoints(n_points, include_normals=True)]
    if augment_rot_z:
        transform_lst.append(RandomApply(T.RandomRotate(360, axis=2), p=0.3))
    if augment_jitter:
        transform_lst.append(RandomApply(T.RandomJitter(0.01), p=0.3))

    return GrabObjPcdDataset(
        root=cache_dir,
        file_list=file_list,
        transform=T.Compose(transform_lst),
        pre_transform=T.Compose(pre_transform_lst),
        force_reload=not use_cache,
    )


def pyg_collate_wrapper(data_list: List[PyGData]) -> PyGBatch:
    batch_obj = PyGBatch.from_data_list(data_list)
    return batch_obj


if __name__ == "__main__":
    dataset = get_grab_point_cloud_dataset(GRAB_DATA_PATH)
    print(dataset)

    for i in range(len(dataset)):
        dp = dataset[i]
        print(dp)
        break

    batch_size = 4
    data_list = [dataset[i] for i in range(batch_size)]
    data = pyg_collate_wrapper(data_list)
    print(data.batch.shape)
    print(data.normal.shape)
    print(data.pos.shape)
