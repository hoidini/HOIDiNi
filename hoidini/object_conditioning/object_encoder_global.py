import numpy as np
import torch
import time
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.data import Data as PyGData
from hoidini.object_conditioning.object_pointcloud_dataset import (
    get_grab_point_cloud_dataset,
    pyg_collate_wrapper,
)
from hoidini.resource_paths import GRAB_DATA_PATH
from bps_torch.bps import bps_torch


if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class ObjectEncoder(torch.nn.Module):
    def __init__(self, n_points: int = 1024, out_dim: int = 32):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([n_points, 512, 256, out_dim], dropout=0.5, norm=None)

    def forward(self, data: PyGData):
        sa0_out = (data.x, data.pos, data.batch)
        # if data.x is not None:
        # sa0_out = (data.x, data.pos, data.batch)
        # elif data.normal is not None:
        # sa0_out = (data.normal, data.pos, data.batch)
        # else:
        # raise ValueError("Data object must have either 'x' or 'normal' attribute")
        # sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        embd = self.mlp(x)
        return embd


class BpsMlp(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        bps_basis_path: str,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.mlp = MLP(
            [in_dim, *[hidden_dim] * hidden_layers, out_dim], dropout=0.5, norm=None
        )

        d = np.load(bps_basis_path)
        self.bps_basis = torch.from_numpy(d["basis"]).float()

    def forward(self, data: PyGData):
        points = data.pos  # (B, n_points, 3)
        dists = torch.cdist(points, self.bps_basis)  # (B, n_points, n_basis)
        dists = torch.min(dists, dim=2).values  # (B, n_points)
        x = self.mlp(dists)
        return x


def main_bps_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initiate the bps module
    bps = bps_torch(
        bps_type="random_uniform",
        n_bps_points=1024,
        radius=1.0,
        n_dims=3,
        custom_basis=None,
    )

    pointcloud = torch.rand([1000000, 3]).to(device)

    s = time.time()

    bps_enc = bps.encode(
        pointcloud, feature_type=["dists", "deltas"], x_features=None, custom_basis=None
    )

    print(time.time() - s)
    deltas = bps_enc["deltas"]
    bps_dec = bps.decode(deltas)
    print(bps_dec.shape)


def main():
    dataset = get_grab_point_cloud_dataset(GRAB_DATA_PATH)
    data_list = [dataset[0], dataset[1], dataset[2]]
    batch = pyg_collate_wrapper(data_list)
    encoder = ObjectEncoder()
    embd = encoder(batch)
    print(embd.shape)


if __name__ == "__main__":
    # main()
    main_bps_mlp()
