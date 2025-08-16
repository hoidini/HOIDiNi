from typing import Dict, Any
import torch
import hydra
import numpy as np
from einops import rearrange
import random
import os
import json
import shutil
import time
import logging
import sys
import colorsys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
TMP_DIR = os.environ.get("TMP_DIR", "/home/dcor/roeyron/tmp/")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def transform_points(x, mat):
    shape = x.shape
    x = rearrange(x, "b t (j c) -> b (t j) c", c=3)  # B x N x 3
    x = torch.einsum(
        "bpc,bck->bpk", mat[:, :3, :3], x.permute(0, 2, 1)
    )  # B x 3 x N   N x B x 3
    x = x.permute(2, 0, 1) + mat[:, :3, 3]
    x = x.permute(1, 0, 2)
    x = x.reshape(shape)

    return x


def create_meshgrid(bbox, size, batch_size=1):
    x = torch.linspace(bbox[0], bbox[1], size[0])
    y = torch.linspace(bbox[2], bbox[3], size[1])
    z = torch.linspace(bbox[4], bbox[5], size[2])
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    grid = grid.repeat(batch_size, 1, 1)

    # aug_z = 0.75 + torch.rand(batch_size, 1) * 0.35
    # grid[:, :, 2] = grid[:, :, 2] * aug_z

    return grid


def rigid_transform_3D(A, B, scale=False):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        # return None, None, None
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t


def find_free_port():
    from contextlib import closing
    import socket

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def init_model(model_cfg, device, eval, load_state_dict=False):
    model = hydra.utils.instantiate(model_cfg)
    if eval:
        load_state_dict_eval(model, model_cfg.ckpt, device=device)
    else:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        if load_state_dict:
            model.module.load_state_dict(torch.load(model_cfg.ckpt))
            model.train()

    return model


def load_state_dict_eval(model, state_dict_path, map_location="cuda:0", device="cuda"):
    state_dict = torch.load(state_dict_path, map_location=map_location)
    key_list = [key for key in state_dict.keys()]
    for old_key in key_list:
        new_key = old_key.replace("module.", "")
        state_dict[new_key] = state_dict.pop(old_key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


def zup_to_yup(coord):
    if len(coord.shape) > 1:
        coord = coord[..., [0, 2, 1]]
        coord[..., 2] *= -1
    else:
        coord = coord[[0, 2, 1]]
        coord[2] *= -1

    return coord


def get_least_busy_device() -> torch.device:
    if not torch.cuda.is_available():
        print("No GPUs are available. Defaulting to CPU.")
        return torch.device("cpu")

    least_busy_gpu = None
    max_free_memory = 0

    for i in range(torch.cuda.device_count()):
        # Get memory stats for each GPU
        stats = torch.cuda.mem_get_info(i)
        free_memory, _ = stats

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            least_busy_gpu = i

    if least_busy_gpu is not None:
        return torch.device(f"cuda:{least_busy_gpu}")
    else:
        print("Could not determine the least busy GPU. Defaulting to CPU.")
        return torch.device("cpu")


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def torchify_numpy_dict(
    d: Dict[str, np.ndarray], device=None, dtype=None
) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in d.items():
        if isinstance(v, list):
            v = np.array(v)
        elif isinstance(v, np.ndarray):
            v = torch.from_numpy(v).to(device, dtype)
        result[k] = v
    return result


def numpify_torch_dict(d: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        result[k] = v
    return result


def torchify_numpy_dict_recursive(
    d: Dict[str, Any], device=None, dtype=None
) -> Dict[str, Any]:
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = torchify_numpy_dict_recursive(v, device, dtype)
        elif isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v).to(device, dtype)
        else:
            result[k] = v
    return result


def create_new_dir(path):
    path = str(path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


class Timer:
    """
    timer = Timer()

    timer.tic("Load Data")
    # Simulate loading data
    time.sleep(0.5)
    timer.tac()  # Stop "Load Data" timer

    timer.tic("Process Data")
    # Simulate processing data
    time.sleep(0.75)
    timer.tac()  # Stop "Process Data" timer

    timer.report()
    """

    def __init__(self):
        self.records = []  # List to store (label, elapsed time) tuples.
        self._stack = []  # Stack to store (label, start time) tuples.

    def tic(self, label: str):
        """Start a timer with the given label."""
        self._stack.append((label, time.perf_counter()))

    def tac(self) -> float:
        """
        Stop the most recently started timer, record the elapsed time,
        and return the elapsed time.
        """
        if not self._stack:
            raise ValueError("No active timer to stop. Please call tic() first.")
        label, start = self._stack.pop()
        elapsed = time.perf_counter() - start
        self.records.append((label, elapsed))
        return elapsed

    def report(self):
        """Prints a report of all timed segments."""
        print("Timing Report:")
        for label, elapsed in self.records:
            print(f" - {label}: {elapsed:.6f} seconds")


def get_logger(name=__name__):
    """
    from general_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Hello, world!")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s-%(filename)s:%(lineno)d - %(levelname)s - %(message)s",
            datefmt="%d:%H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def get_model_identifier(model_path: str):
    """
    /a/b/c/experiment_name/model0010.pt --> experiment_name__model0010
    """
    experiment_dir = os.path.dirname(os.path.dirname(model_path))
    exp_slash_model = os.path.relpath(model_path, experiment_dir)
    model_identifier = exp_slash_model.replace(".pt", "").replace("/", "__")
    return model_identifier


def get_distinguishable_rgba_colors(n: int, alpha: float = 1.0):
    """
    Generate n visually distinguishable RGBA colors using HSV space.

    Args:
        n (int): Number of colors to generate.
        alpha (float): Alpha channel value for all colors (0.0–1.0).

    Returns:
        List[Tuple[float, float, float, float]]: List of RGBA colors.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b, alpha))
    return colors


def reshape_mdm_features_to_standard_format(features: torch.Tensor) -> torch.Tensor:
    """
    Reshapes MDM (Motion Diffusion Model) features from shape (batch, channels, 1, seq_len)
    to standard format (batch, seq_len, channels) for easier processing.

    Args:
        features: Tensor of shape (batch, channels, 1, seq_len)

    Returns:
        Tensor of shape (batch, seq_len, channels)
    """
    assert features.dim() == 4
    assert features.shape[2] == 1
    return features.permute(0, 2, 3, 1).squeeze()  # (batch, seq_len, n_features)


def reshape_standard_features_to_mdm_format(features: torch.Tensor) -> torch.Tensor:
    """
    Reshapes standard features (batch, seq_len, channels) to MDM (Motion Diffusion Model) format (batch, channels, 1, seq_len).

    Args:
        features: Tensor of shape (batch, seq_len, channels)

    Returns:
        Tensor of shape (batch, channels, 1, seq_len)
    """
    assert features.dim() == 3
    return features.permute(0, 2, 1).unsqueeze(2)  # (batch, channels, 1, seq_len)
