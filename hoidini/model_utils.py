import torch
import torch_geometric


def model_kwargs_to_device(model_kwargs, device):
    model_kwargs["y"] = {
        key: (
            val.to(device)
            if torch.is_tensor(val) or isinstance(val, torch_geometric.data.Data)
            else val
        )
        for key, val in model_kwargs["y"].items()
    }
