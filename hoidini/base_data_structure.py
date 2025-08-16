from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Dict, Self
import numpy as np
import torch

from hoidini.general_utils import torchify_numpy_dict


@dataclass
class SequentialData(ABC):

    def to_dict(self) -> Dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def get_sequential_attr_names(cls):
        """
        Returns a list of the sequential elements of the contact record.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_spatial_attr_names(cls):
        raise NotImplementedError("Subclasses must implement this method")

    def cut(self, start_frame: int, end_frame: int) -> Self:
        d = self.to_dict()
        seq_attr_names = self.get_sequential_attr_names()
        d = {
            k: v[start_frame:end_frame] if k in seq_attr_names else v
            for k, v in d.items()
        }
        return self.__class__(**d)

    def extend(self, n_frames: int) -> Self:
        """Extend sequence to reach a total length of n_frames by repeating the last frame"""
        d = self.to_dict()
        seq_attr_names = self.get_sequential_attr_names()

        # Get current length from first sequential attribute
        if seq_attr_names:
            first_attr = seq_attr_names[0]
            current_length = len(d[first_attr])
            # Only extend if target length is greater than current length
            if n_frames > current_length:
                frames_to_add = n_frames - current_length
                for attr_name in seq_attr_names:
                    last_frame = d[attr_name][-1:]  # Get the last frame
                    repeated_frames = last_frame.repeat(
                        frames_to_add, *([1] * (d[attr_name].dim() - 1))
                    )  # Repeat it frames_to_add times
                    d[attr_name] = torch.cat([d[attr_name], repeated_frames], dim=0)

        return self.__class__(**d)

    def translate(self, translation: torch.Tensor) -> Self:
        d = self.to_dict()
        translation = translation.flatten()
        spatial_attr_names = self.get_spatial_attr_names()
        d = {
            k: v + translation.to(v.device) if k in spatial_attr_names else v
            for k, v in d.items()
        }
        return self.__class__(**d)

    def clone(self) -> Self:
        d = self.to_dict()
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.clone()
            else:
                d[k] = deepcopy(v)
        return self.__class__(**d)

    def detach(self) -> Self:
        d = self.to_dict()
        for key, value in d.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                d[key] = value
        return self.__class__(**d)

    def to(self, *args, **kwargs) -> Self:
        d = self.to_dict()
        for key, value in d.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.to(*args, **kwargs)
                d[key] = value
        return self.__class__(**d)

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float):
        assert path.endswith(".npz"), "File must be a .npz file"
        data = np.load(path, allow_pickle=True)
        if "data_dict" in data:
            d = data["data_dict"].item()
        else:
            # Handle older saved files
            d = dict(data)
        d = torchify_numpy_dict(d, device, dtype)
        return cls(**d)

    def save(self, path):
        d = self.to_dict()
        np.savez(path, data_dict=d, allow_pickle=True)


def test_torch_dataclass():
    """Test the TorchDataclass class functionality"""

    @dataclass
    class TestDataStructure(SequentialData):
        seq1: torch.Tensor
        seq2: torch.Tensor
        some_string: str

    test_data_structure = TestDataStructure(
        seq1=torch.randn(10, 10), seq2=torch.randn(10, 10), some_string="test"
    )
    test_data_structure.save("test.npz")

    test_data_structure_loaded = TestDataStructure.load("test.npz")
    print(test_data_structure_loaded)

    assert (
        test_data_structure.to_dict().keys()
        == test_data_structure_loaded.to_dict().keys()
    )
    for key in test_data_structure.to_dict().keys():
        if isinstance(test_data_structure.to_dict()[key], torch.Tensor):
            assert torch.allclose(
                test_data_structure.to_dict()[key],
                test_data_structure_loaded.to_dict()[key],
            )
        else:
            assert (
                test_data_structure.to_dict()[key]
                == test_data_structure_loaded.to_dict()[key]
            )
    print("Test passed")


if __name__ == "__main__":
    test_torch_dataclass()
