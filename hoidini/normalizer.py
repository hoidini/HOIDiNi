from __future__ import annotations
import os
import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """
    Standardizes inputs:  x ↦ (x - mean) / std
    * If std < eps for a channel (i.e. constant feature) we clamp std to1,
      so the feature is only centred (not scaled) and we avoid NaNs/Inf.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-5,
        disable: bool = False,
    ):
        super().__init__()

        # ─── safe std: clamp tiny values ────────────────────────────────────────
        safe_std = std.clone()
        safe_std[safe_std < eps] = 1.0  # “no‑scale” for constant features

        if (safe_std < eps).any():
            print(f"Warning, std smaller than eps: {safe_std < eps}")

        # ─── register buffers ──────────────────────────────────────────────────
        self.register_buffer("mean", mean)
        self.register_buffer("std", safe_std)
        self.eps = eps
        self.disable = disable

    @property
    def n_feats(self):
        return self.mean.shape[0]

    # -------------------------------------------------------------------------
    #  I/O helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def exists(base_dir: str, split: str = "") -> bool:
        return all(
            os.path.exists(Normalizer._get_path(obj, base_dir, split))
            for obj in ("mean", "std", "eps")
        )

    @classmethod
    def from_dir(
        cls, base_dir: str, split: str = "", device=None, disable: bool = False
    ):
        mean = torch.load(cls._get_path("mean", base_dir, split)).to(device)
        std = torch.load(cls._get_path("std", base_dir, split)).to(device)
        eps = torch.load(cls._get_path("eps", base_dir, split))
        return cls(mean, std, eps, disable)

    @staticmethod
    def _get_path(obj_name: str, base_dir: str, split: str = "") -> str:
        fname = f"{obj_name}.pt" if split == "" else f"{obj_name}_{split}.pt"
        return os.path.join(base_dir, fname)

    def save(self, base_dir: str, split: str = "") -> None:
        os.makedirs(base_dir, exist_ok=True)
        torch.save(self.mean, self._get_path("mean", base_dir, split))
        torch.save(self.std, self._get_path("std", base_dir, split))
        torch.save(self.eps, self._get_path("eps", base_dir, split))

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable:
            return x
        return (x - self.mean) / self.std  # std already “safe”

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable:
            return x
        return x * self.std + self.mean
