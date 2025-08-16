import os
from glob import glob
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from hoidini.datasets.dataset_smplrifke import load_hml_split_ids
import smplx
import torch
from hoidini.blender_utils.visualize_stick_figure_blender import visualize_motions
from hoidini.general_utils import torchify_numpy_dict
from hoidini.skeletons.smplx_52 import SMPLX_52_144_INDS
from hoidini.datasets.smpldata import SMPL_MODELS_PATH, SmplData

AMASS_PATH = (
    "/home/dcor/roeyron/trumans_utils/DATASETS/AMASS_SMPL-X/AMASS_SMPL-X_extracted"
)

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


def get_df_index_hml3d():
    df_hml3d = pd.read_csv(
        "/home/dcor/roeyron/trumans_utils/src/datasets/resources/humanml3d_index.csv"
    )
    df_hml3d["dataset"] = df_hml3d.source_path.apply(lambda p: p.split("/")[2])
    df_hml3d["dp_name"] = df_hml3d["new_name"].apply(
        lambda p: p.split("/")[-1].replace(".npz", "").replace(".npy", "")
    )
    return df_hml3d


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


def map_datapoint_path(path, amass_path, dataset):
    path = path.replace("./pose_data/", "")
    path = os.path.relpath(path, os.path.dirname(os.path.dirname(path)))
    path = os.path.join(amass_path, HML_TO_AMASS_NAME_MAP.get(dataset, dataset), path)
    path = path.replace("poses.npy", "stageii.npz")
    path = path.replace(" ", "_")
    return path


def get_hml3d_extended_smpldata_lst(split=None, lim=False, fk_device="cpu"):
    df_index_hml3d = get_df_index_hml3d()
    if split is not None:
        hml_split_ids = set(load_hml_split_ids(split))
        df_index_hml3d = df_index_hml3d[
            df_index_hml3d.dp_name.apply(lambda dp_name: dp_name in hml_split_ids)
        ]
    print("#samples =", len(df_index_hml3d))
    print("#datasets =", len(df_index_hml3d.dataset.unique()), "\n")
    count_hml3d = dict(
        df_index_hml3d.groupby("dataset").size()
    )  # .reset_index(name='count')
    pprint(count_hml3d)

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

    amass_gender_per_dataset = {
        row.dataset: row.gender for _, row in df_amass.iterrows()
    }

    datasets_to_use = list(
        set(count_amass.keys()).intersection(set(count_hml3d.keys()))
    )
    amass_gender_per_dataset = {
        row.dataset: row.gender for _, row in df_amass.iterrows()
    }
    for dataset in datasets_to_use:
        assert amass_gender_per_dataset[dataset] == "neutral"

    print("#datasts", len(datasets_to_use))
    print()
    pprint(datasets_to_use)
    print()
    print("--- datasets from HML3D that won't be used:")
    ignore_datasets = [
        dataset for dataset in count_hml3d if dataset not in datasets_to_use
    ]
    pprint(ignore_datasets)

    # # Build df_index

    df_index = df_index_hml3d.copy()
    df_index = df_index[
        df_index.dataset.apply(lambda dataset: dataset not in ignore_datasets)
    ]
    print(len(df_index.dataset.unique()))

    df_index["smplx_path"] = [
        map_datapoint_path(row.source_path, AMASS_PATH, row.dataset)
        for _, row in df_index.iterrows()
    ]
    df_index.head()
    ext_smpldata_lst = []
    for ind, row in tqdm(df_index.iterrows(), total=len(df_index), disable=False):
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

        path = row.smplx_path
        if not os.path.exists(path):
            print(f"SKIPPED {path} because it doesn't exist")
            continue
        data = dict(np.load(path, allow_pickle=True))
        mocap_fps = data["mocap_frame_rate"]
        tgt_fps = 20
        if (int(mocap_fps) % tgt_fps) != 0:
            print(f"mocap_fps={mocap_fps} is not devisable by tgt_fps={tgt_fps}", row)

        skip_frames = round(mocap_fps / tgt_fps)

        # frame_selection (reduce fps + cut)
        start_frame = row.start_frame
        end_frame = row.end_frame
        for k in ["trans", "root_orient", "pose_body", "pose_hand", "poses"]:
            arr = data[k]
            arr = arr[::skip_frames]
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

        sbj_model: smplx.SMPLX = smplx.create(
            model_path=SMPL_MODELS_PATH,
            model_type="smplx",
            gender="neutral",
            create_jaw_pose=True,
            batch_size=T,
            use_pca=False,
            flat_hand_mean=True,
        ).to(fk_device)

        smplx_input = torchify_numpy_dict(
            smplx_input, device=torch.device("cpu"), dtype=torch.float
        )
        with torch.no_grad():
            smplx_output = sbj_model(**smplx_input)

        joints = smplx_output.joints[:, SMPLX_52_144_INDS]
        poses = torch.concat(
            [
                smplx_input["global_orient"],
                smplx_input["body_pose"],
                smplx_input["left_hand_pose"],
                smplx_input["right_hand_pose"],
            ],
            dim=1,
        )

        smpldata = SmplData(
            poses=poses,
            trans=smplx_input["transl"],
            joints=joints,
        )
        ext_smpldata_lst.append(
            {
                "smpldata": smpldata,
                "source_path": row.source_path,
                "dataset": row.dataset,
                "smplx_path": row.smplx_path,
                "hml3d_name": row.new_name.replace(".npz", "").replace(".npy", ""),
            }
        )

        if lim is not None and len(ext_smpldata_lst) >= lim:
            break
    return ext_smpldata_lst


def main():
    ext_smpldata_lst = get_hml3d_extended_smpldata_lst(lim=3)
    for ext_smpldata in ext_smpldata_lst:
        print(ext_smpldata["smpldata"].poses.shape)
        print(ext_smpldata["smpldata"].trans.shape)


if __name__ == "__main__":
    main()
