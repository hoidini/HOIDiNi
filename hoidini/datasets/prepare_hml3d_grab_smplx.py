import os
from glob import glob
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import smplx
import torch
from hoidini.blender_utils.visualize_stick_figure_blender import visualize_motions
from hoidini.general_utils import torchify_numpy_dict
from hoidini.skeletons.smplx_52 import SMPLX_52_144_INDS
from hoidini.datasets.smpldata import SMPL_MODELS_PATH
from hoidini.general_utils import get_least_busy_device, create_new_dir

device = get_least_busy_device()

AMASS_PATH = (
    "/home/dcor/roeyron/trumans_utils/DATASETS/AMASS_SMPL-X/AMASS_SMPL-X_extracted"
)
debug = False
output_path = (
    "/home/dcor/roeyron/trumans_utils/DATASETS/Data_SMPLX_HML3D_GRAB_smplrifke_inputs"
)
# create_new_dir(output_path)


AMASS_TO_HML_NAME_MAP = {
    # amass_name: hml_name
    "DFaust": "DFaust_67",
    "Transitions": "Transitions_mocap",
    "SSM": "SSM_synced",
    "HDM05": "MPI_HDM05",
    "MoSh": "MPI_mosh",
    "PosePrior": "MPI_Limits",
    "BMLrub": "BioMotionLab_NTroje",
}

HML_TO_AMASS_NAME_MAP = {v: k for k, v in AMASS_TO_HML_NAME_MAP.items()}

# # Load datasets motions DataFrames
# ### HumanML3D


def get_df_index_hml3d():
    df_hml3d = pd.read_csv(
        "/home/dcor/roeyron/trumans_utils/src/datasets/resources/humanml3d_index.csv"
    )
    df_hml3d["dataset"] = df_hml3d.source_path.apply(lambda p: p.split("/")[2])
    return df_hml3d


df_index_hml3d = get_df_index_hml3d()
print("#samples =", len(df_index_hml3d))
print("#datasets =", len(df_index_hml3d.dataset.unique()), "\n")
count_hml3d = dict(
    df_index_hml3d.groupby("dataset").size()
)  # .reset_index(name='count')
pprint(count_hml3d)
df_index_hml3d

# ### AMASS


def load_df_amass():
    sub_dataset_names = os.listdir(AMASS_PATH)

    motion_paths = glob("**/*.npz", root_dir=AMASS_PATH, recursive=True)
    motion_paths = [os.path.join(AMASS_PATH, mp) for mp in motion_paths]
    ddl = defaultdict(list)
    # print('total #dataset =', len(sub_dataset_names))
    for dataset in sub_dataset_names:
        sub_dataset_path = os.path.join(AMASS_PATH, dataset)
        motion_rel_paths = glob("**/*.npz", root_dir=sub_dataset_path, recursive=True)
        motion_paths = [os.path.join(sub_dataset_path, rp) for rp in motion_rel_paths]
        sample_path = motion_paths[0]
        data = dict(np.load(sample_path, allow_pickle=True))

        dataset_name = AMASS_TO_HML_NAME_MAP.get(dataset, dataset)
        ddl["dataset"].append(dataset_name)
        ddl["gender"].append(data["gender"])
        ddl["n_samples"].append(len(motion_paths))
        # ddl['n_moving_joints'].append(np.sum(data['poses'].std(axis=0) > 1e-6) / 3)
    df = pd.DataFrame(ddl)
    return df


df_amass = load_df_amass()
print(df_amass.n_samples.sum())
count_amass = {row.dataset: row.n_samples for _, row in df_amass.iterrows()}
pprint(count_amass)
df_amass

# # Compare datasets

only_amass = [k for k in count_amass if k not in count_hml3d]
only_hml3d = [k for k in count_hml3d if k not in count_amass]
print("Only in AMASS:", only_amass)
print("Only in HumanML3d:", only_hml3d)

amass_gender_per_dataset = {row.dataset: row.gender for _, row in df_amass.iterrows()}

datasets_to_use = list(set(count_amass.keys()).intersection(set(count_hml3d.keys())))
amass_gender_per_dataset = {row.dataset: row.gender for _, row in df_amass.iterrows()}
for dataset in datasets_to_use:
    assert amass_gender_per_dataset[dataset] == "neutral"

print("#datasts", len(datasets_to_use))
print()
pprint(datasets_to_use)
print()
print("--- datasets from HML3D that won't be used:")
ignore_datasets = [dataset for dataset in count_hml3d if dataset not in datasets_to_use]
pprint(ignore_datasets)

# # Build df_index

df_index = df_index_hml3d.copy()
df_index = df_index[
    df_index.dataset.apply(lambda dataset: dataset not in ignore_datasets)
]
print(len(df_index.dataset.unique()))
df_index.head()

# ### Fix motions path (HML3D to AMASS)


def map_datapoint_path(path, amass_path, dataset):
    path = path.replace("./pose_data/", "")
    path = os.path.relpath(path, os.path.dirname(os.path.dirname(path)))
    path = os.path.join(amass_path, HML_TO_AMASS_NAME_MAP.get(dataset, dataset), path)
    path = path.replace("poses.npy", "stageii.npz")
    path = path.replace(" ", "_")
    return path


df_index["smplx_path"] = [
    map_datapoint_path(row.source_path, AMASS_PATH, row.dataset)
    for _, row in df_index.iterrows()
]
df_index.head()

# ### ADD GRAB

grab_path = os.path.join(AMASS_PATH, "GRAB")
motion_names = glob("**/*.npz", root_dir=grab_path, recursive=True)
ddl = defaultdict(list)
for motion_name in motion_names:
    path = os.path.join(grab_path, motion_name)
    ddl["source_path"].append(motion_name)
    ddl["start_frame"].append(None)
    ddl["end_frame"].append(None)
    ddl["dataset"].append("GRAB")
    ddl["new_name"].append(motion_name.replace("_stageii.npz", ".npz"))
    ddl["smplx_path"].append(path)
df_index_grab = pd.DataFrame(ddl)
df_index_grab

df_index = pd.concat([df_index, df_index_grab])
df_index.head()

# # Extract SMPL-X poses


if debug:
    df_index_to_use = df_index.sort_values("dataset").groupby("dataset").head(2)
    blend_dir_path = "/home/dcor/roeyron/tmp/amass_blend_vis"
    create_new_dir(blend_dir_path)
else:
    df_index_to_use = df_index


df_index.to_csv(os.path.join(output_path, "index.csv"))

missing = []
for ind, row in tqdm(
    df_index_to_use.iterrows(), total=len(df_index_to_use), disable=False
):

    new_path = os.path.join(output_path, row.new_name.replace(".npy", ".npz"))

    if row.dataset == "GRAB" and os.path.exists(new_path):
        os.remove(new_path)

    if os.path.exists(new_path):
        continue
    if not os.path.exists(row.smplx_path):
        missing.append(row.smplx_path)
        print("MISSING", row.smplx_path)
        continue

    to_skip_lst = [
        "/neutral_stagei.npz",
    ]
    skip = False
    for to_skip in to_skip_lst:
        if to_skip in row.smplx_path:
            print("SKIPPED", row.smplx_path)
            skip = True
            break
    if skip:
        continue

    data = dict(np.load(row.smplx_path, allow_pickle=True))

    mocap_fps = data["mocap_frame_rate"]
    tgt_fps = 20

    if (int(mocap_fps) % tgt_fps) != 0:
        print(f"mocap_fps={mocap_fps} is not devisable by tgt_fps={tgt_fps}", path)

    skip_frames = round(mocap_fps / tgt_fps)

    # frame_selection (reduce fps + cut)
    start_frame = row.start_frame
    end_frame = row.end_frame
    for k in ["trans", "root_orient", "pose_body", "pose_hand", "poses"]:
        arr = data[k]
        arr = arr[::skip_frames]
        if row.dataset != "GRAB":  # GRAB will be trimmed inside the dataset
            arr = arr[start_frame:end_frame]
        data[k] = arr

    smplx_input = {
        "transl": data["trans"],
        "global_orient": data["root_orient"],
        "body_pose": data["pose_body"],
        "left_hand_pose": data["pose_hand"][:, : 15 * 3],
        "right_hand_pose": data["pose_hand"][:, 15 * 3 :],
    }
    T = smplx_input["body_pose"].shape[0]
    if T <= 5:
        continue

    smplx_input = torchify_numpy_dict(smplx_input, device, torch.float)

    sbj_model: smplx.SMPLX = smplx.create(
        model_path=SMPL_MODELS_PATH,
        model_type="smplx",
        gender="neutral",
        create_jaw_pose=True,
        batch_size=T,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    smplx_output: smplx.SMPL.forward = sbj_model(**smplx_input)

    joints = smplx_output.joints[:, SMPLX_52_144_INDS]

    if debug:
        save_path = os.path.join(blend_dir_path, f"{row.dataset}_{ind}.blend")
        visualize_motions([joints[:200]], save_path, print_download_and_run=False)

    smpl_data_dict = {
        "poses": np.concatenate(
            [data["root_orient"], data["pose_body"], data["pose_hand"]], axis=1
        ),
        "trans": data["trans"],
        "joints": joints.detach().cpu().numpy(),
    }

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    np.savez(new_path, **smpl_data_dict)


assert len(missing) <= 107
