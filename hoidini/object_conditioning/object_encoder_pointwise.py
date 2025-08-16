"""
Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
"""

from typing import List
import torch
from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
import torch_geometric

from hoidini.object_conditioning.object_encoder_global import GlobalSAModule, SAModule
from hoidini.object_conditioning.object_pointcloud_dataset import (
    get_grab_point_cloud_dataset,
    pyg_collate_wrapper,
)
from hoidini.resource_paths import GRAB_DATA_PATH

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class ObjectPointwiseEncoder(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, out_dim]))

    def forward(self, data):
        if data.x is not None:
            sa0_out = (data.x, data.pos, data.batch)
        elif data.normal is not None:
            sa0_out = (data.normal, data.pos, data.batch)
        else:
            raise ValueError("Data object must have either 'x' or 'normal' attribute")
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        # split to batches
        x = torch.stack(
            torch.split(x, (data.ptr[1:] - data.ptr[:-1]).tolist(), dim=0), dim=0
        )
        return x


def pyg_batch_to_torch_batch(
    data: torch_geometric.data.Batch, attr: str
) -> torch.Tensor:
    x = torch.split(getattr(data, attr), (data.ptr[1:] - data.ptr[:-1]).tolist(), dim=0)
    return torch.stack(x, dim=0)


def main():
    dataset = get_grab_point_cloud_dataset(GRAB_DATA_PATH, use_cache=False)
    data_list = [dataset[0], dataset[1], dataset[2]]
    batch = pyg_collate_wrapper(data_list)

    print("Batch normal shape:", batch.normal.shape)
    print("Batch pos shape:", batch.pos.shape)
    print("Batch batch shape:", batch.batch.shape)
    encoder = ObjectPointwiseEncoder(512)
    print("encoder #params (M):", sum(p.numel() for p in encoder.parameters()) / 1e6)
    embd = encoder(batch)
    print(embd.shape)


if __name__ == "__main__":
    main()
