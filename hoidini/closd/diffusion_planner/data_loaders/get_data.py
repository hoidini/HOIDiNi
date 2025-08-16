from typing import Optional
from functools import partial
from torch.utils.data import DataLoader
import os
import numpy as np

from hoidini.closd.diffusion_planner.data_loaders.tensors import collate as all_collate
from hoidini.closd.diffusion_planner.data_loaders.tensors import (
    t2m_collate,
    t2m_prefix_collate,
)
from hoidini.closd.utils import hf_handler
from hoidini.datasets.dataset_smplrifke import (
    GrabRifkeDataset,
    HML3DSmplRifkeDataset,
    Hml3DGrabDataset,
    HoiGrabDataset,
    collate_smplrifke_mdm,
    get_mean_and_std,
)
from hoidini.normalizer import Normalizer
from hoidini.general_utils import SRC_DIR
from hoidini.closd.diffusion_planner.utils.fixseed import fixseed


def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS

        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC

        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses

        return HumanAct12Poses
    elif name == "humanml":
        from hoidini.closd.diffusion_planner.data_loaders.humanml.data.dataset import (
            HumanML3D,
        )

        return HumanML3D
    elif name == "kit":
        from hoidini.closd.diffusion_planner.data_loaders.humanml.data.dataset import (
            KIT,
        )

        return KIT
    else:
        raise ValueError(f"Unsupported dataset name [{name}]")


def get_collate_fn(name, hml_mode="train", pred_len=0, batch_size=1):
    if hml_mode == "gt":
        from hoidini.closd.diffusion_planner.data_loaders.humanml.data.dataset import (
            collate_fn as t2m_eval_collate,
        )

        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(
    name,
    num_frames=None,
    split="train",
    hml_mode="train",
    abs_path=".",
    fixed_len=0,
    hml_type=None,
    device=None,
    autoregressive=False,
    return_keys=False,
    cache_path=None,
):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(
            split=split,
            num_frames=num_frames,
            mode=hml_mode,
            abs_path=abs_path,
            fixed_len=fixed_len,
            hml_type=hml_type,
            device=device,
            autoregressive=autoregressive,
            return_keys=return_keys,
            cache_path=cache_path,
        )
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def worker_init_fn(id, seed):
    if seed is not None:
        fixseed(id + seed)
    else:
        fixseed(id + np.random.randint(1000000))


def get_dataset_loader(
    name,
    batch_size,
    num_frames=None,
    split="train",
    hml_mode="train",
    hml_type=None,
    fixed_len=0,
    pred_len=0,
    device=None,
    autoregressive=False,
    drop_last=True,
    return_keys=False,
    data_load_lim: Optional[float] = None,
    experiment_dir=None,
    is_training=False,
    dataset_data=None,
    seed=None,
    use_cache=False,
    features_string=None,
    grab_seq_paths=None,
    grab_split="train",
    condition_input_feature=None,
):
    sampler = None
    if name == "smplrifke":
        if dataset_data == "hml3d":
            dataset_path = (
                "/home/dcor/roeyron/trumans_utils/DATASETS/Data_AMASS_smplrifke_inputs"
            )
            dataset = HML3DSmplRifkeDataset(
                dataset_path, seq_len=fixed_len, lim=data_load_lim, split=split
            )
        elif dataset_data == "grab":
            dataset_path = (
                "/home/dcor/roeyron/trumans_utils/DATASETS/Data_GRAB_smplrifke_inputs"
            )
            dataset = GrabRifkeDataset(
                dataset_path, seq_len=fixed_len, lim=data_load_lim
            )
        elif dataset_data == "hml3d_grab":
            dataset_path = "/home/dcor/roeyron/trumans_utils/DATASETS/Data_SMPLX_HML3D_GRAB_smplrifke_inputs"
            dataset = Hml3DGrabDataset(
                dataset_path,
                seq_len=fixed_len,
                lim=data_load_lim,
                hml_split=split,
                fk_device="cpu",
                grab_seq_paths=grab_seq_paths,
                grab_split=grab_split,
                features_string=features_string,
                condition_input_feature=condition_input_feature,
            )
            # dataset = Hml3DGrabDataset(dataset_path, seq_len=fixed_len, lim=debug_limit, hml_split=split, fk_device=dist_util.dev())
            sampler = dataset.get_sampler(out_lim_factor=10)
        elif dataset_data == "grabhoi":
            dataset_path = (
                "/home/dcor/roeyron/trumans_utils/DATASETS/DATA_GRAB_RETARGETED"
            )
            dataset = HoiGrabDataset(
                dataset_path,
                seq_len=fixed_len,
                lim=data_load_lim,
                use_cache=use_cache,
                features_string=features_string,
                grab_seq_paths=grab_seq_paths,
                grab_split=grab_split,
            )
        else:
            raise ValueError(f"Unsupported dataset_data: {dataset_data}")

        if is_training:
            mean, std = get_mean_and_std(dataset, sample_size=2000)
            normalizer = Normalizer(mean, std)
            normalizer.save(experiment_dir)
        else:
            normalizer = Normalizer.from_dir(experiment_dir)
        dataset.set_normalizer(normalizer)
        collate = partial(
            collate_smplrifke_mdm, pred_len=pred_len, context_len=fixed_len - pred_len
        )
    else:
        abs_path = os.path.join(SRC_DIR, "closd/diffusion_planner")
        cache_path = hf_handler.get_dependencies()
        dataset = get_dataset(
            name,
            num_frames,
            split,
            hml_mode,
            abs_path,
            fixed_len,
            hml_type,
            device,
            autoregressive,
            return_keys,
            cache_path,
        )
        collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    if grab_seq_paths is not None or sampler is not None:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size if not data_load_lim else len(dataset),
        num_workers=8,
        drop_last=drop_last,
        collate_fn=collate,
        shuffle=shuffle,
        sampler=sampler,
        persistent_workers=sampler is not None,
        worker_init_fn=partial(worker_init_fn, seed=seed),
    )
    return loader


def replace_loader(dataset, old_loader):
    collate = old_loader.collate_fn
    batch_size = old_loader.batch_size
    drop_last = old_loader.drop_last
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=drop_last,
        collate_fn=collate,
    )
    del old_loader
    return loader
