from copy import deepcopy
import torch
import smplx
import numpy as np
from tqdm import tqdm
import trimesh
from general_utils import (
    TMP_DIR,
    PROJECT_DIR,
    create_new_dir,
    get_least_busy_device,
    torchify_numpy_dict,
)
from resource_paths import GRAB_DATA_PATH, GRAB_ORIG_DATA_PATH
from datasets.smpldata import SMPL_MODELS_PATH
from datasets.grab.grab_utils import parse_npz
import os
from datasets.grab.grab_utils import get_all_grab_seq_paths

device = get_least_busy_device()

LEFT_HAND_INDICES = list(range(25, 40))
RIGHT_HAND_INDICES = list(range(40, 55))
LEFT_ANKLE_INDICES = [7]
RIGHT_ANKLE_INDICES = [8]
LEFT_FEET_INDICES = [10]
RIGHT_FEET_INDICES = [11]


def optimize_neutral_motion(seq_data, num_iterations=100, lr=1e-2):
    seq_data_new = deepcopy(seq_data)
    seq_data["body"]["params"] = torchify_numpy_dict(
        seq_data["body"]["params"], device=device
    )
    seq_data["lhand"]["params"] = torchify_numpy_dict(
        seq_data["lhand"]["params"], device=device
    )
    seq_data["rhand"]["params"] = torchify_numpy_dict(
        seq_data["rhand"]["params"], device=device
    )

    #############
    # Original model
    #############
    orig_gender = seq_data["gender"]
    batch_size = seq_data["n_frames"]

    sbj_mesh_path = os.path.join(
        GRAB_DATA_PATH, seq_data["gender"], os.path.basename(seq_data["body"]["vtemp"])
    )
    sbj_mesh = trimesh.load(sbj_mesh_path)
    # sbj_mesh_faces = np.array(sbj_mesh.faces)
    sbj_mesh_vertices = np.array(sbj_mesh.vertices)
    # n_comps = seq_data['n_comps']
    src_model = smplx.create(
        model_path=SMPL_MODELS_PATH,
        model_type="smplx",
        gender=orig_gender,
        # num_pca_comps=n_comps,
        flat_hand_mean=True,
        use_pca=False,
        v_template=sbj_mesh_vertices,
        batch_size=batch_size,
    ).to(device)

    orig_params = {
        "global_orient": seq_data["body"]["params"]["global_orient"],
        "body_pose": seq_data["body"]["params"]["body_pose"],
        "left_hand_pose": seq_data["lhand"]["params"]["fullpose"],
        "right_hand_pose": seq_data["rhand"]["params"]["fullpose"],
        "transl": seq_data["body"]["params"]["transl"],
    }
    orig_output = src_model(**orig_params)

    #############
    # Target model
    #############
    tgt_model = smplx.create(
        model_path=SMPL_MODELS_PATH,
        model_type="smplx",
        gender="neutral",
        batch_size=batch_size,
        flat_hand_mean=True,
        use_pca=False,
    ).to(device)

    optimizable_params = {
        k: v.clone().detach().requires_grad_(True) for k, v in orig_params.items()
    }

    # optimizer = torch.optim.LBFGS(list(optimizable_params.values()), lr=lr, max_iter=20, line_search_fn='strong_wolfe')
    optimizer = torch.optim.Adam(list(optimizable_params.values()), lr=lr)

    loss_history = []

    def closure():
        optimizer.zero_grad()

        neutral_output = tgt_model(**optimizable_params)
        loss_joints_lhand = torch.mean(
            (
                neutral_output.joints[:, LEFT_HAND_INDICES, :]
                - orig_output.joints[:, LEFT_HAND_INDICES, :]
            )
            ** 2
        )
        loss_joints_rhand = torch.mean(
            (
                neutral_output.joints[:, RIGHT_HAND_INDICES, :]
                - orig_output.joints[:, RIGHT_HAND_INDICES, :]
            )
            ** 2
        )
        overall_joint_loss = torch.mean(
            (neutral_output.joints - orig_output.joints) ** 2
        )
        loss_pose_body = torch.mean(
            (optimizable_params["body_pose"] - orig_params["body_pose"]) ** 2
        )
        loss_pose_lhand = torch.mean(
            (optimizable_params["left_hand_pose"] - orig_params["left_hand_pose"]) ** 2
        )
        loss_pose_rhand = torch.mean(
            (optimizable_params["right_hand_pose"] - orig_params["right_hand_pose"])
            ** 2
        )

        loss_left_ankle = torch.mean(
            (
                neutral_output.joints[:, LEFT_ANKLE_INDICES, :]
                - orig_output.joints[:, LEFT_ANKLE_INDICES, :]
            )
            ** 2
        )
        loss_right_ankle = torch.mean(
            (
                neutral_output.joints[:, RIGHT_ANKLE_INDICES, :]
                - orig_output.joints[:, RIGHT_ANKLE_INDICES, :]
            )
            ** 2
        )
        loss_left_feet = torch.mean(
            (
                neutral_output.joints[:, LEFT_FEET_INDICES, :]
                - orig_output.joints[:, LEFT_FEET_INDICES, :]
            )
            ** 2
        )
        loss_right_feet = torch.mean(
            (
                neutral_output.joints[:, RIGHT_FEET_INDICES, :]
                - orig_output.joints[:, RIGHT_FEET_INDICES, :]
            )
            ** 2
        )

        # Add temporal continuity loss if we have more than one frame
        if batch_size > 1:
            # Calculate temporal differences for each parameter
            body_pose_diff = (
                optimizable_params["body_pose"][1:]
                - optimizable_params["body_pose"][:-1]
            )
            left_hand_pose_diff = (
                optimizable_params["left_hand_pose"][1:]
                - optimizable_params["left_hand_pose"][:-1]
            )
            right_hand_pose_diff = (
                optimizable_params["right_hand_pose"][1:]
                - optimizable_params["right_hand_pose"][:-1]
            )
            global_orient_diff = (
                optimizable_params["global_orient"][1:]
                - optimizable_params["global_orient"][:-1]
            )
            transl_diff = (
                optimizable_params["transl"][1:] - optimizable_params["transl"][:-1]
            )

            # Compute temporal continuity loss
            temporal_loss = (
                torch.mean(body_pose_diff**2)
                + torch.mean(left_hand_pose_diff**2)
                + torch.mean(right_hand_pose_diff**2)
                + torch.mean(global_orient_diff**2)
                + torch.mean(transl_diff**2)
            )
            alpha_joints_hands = 1
            alpha_joints_all = 0.0001
            alpha_pose = 0.1
            alpha_ankles = 0.05
            alpha_feet = 0.7
            alpha_temporal = 0.003
            loss = (
                alpha_joints_hands * (loss_joints_lhand + loss_joints_rhand)
                + alpha_joints_all * overall_joint_loss
                + alpha_pose * (loss_pose_body + loss_pose_lhand + loss_pose_rhand)
                + alpha_temporal * temporal_loss
                + alpha_ankles * (loss_left_ankle + loss_right_ankle)
                + alpha_feet * (loss_left_feet + loss_right_feet)
            )

        loss.backward(retain_graph=True)
        loss_history.append(loss.item())
        return loss

    for i in range(num_iterations):
        optimizer.step(closure)
        if i == 0:
            print(f"Iteration -1: Loss = {loss_history[0]:.6f}")
        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss_history[-1]:.6f}")

    optimized_params = {k: v.detach() for k, v in optimizable_params.items()}

    del seq_data_new["body"]["params"]["fullpose"]  # no longer correct
    seq_data_new["lhand"]["params"]["fullpose"] = (
        optimized_params["left_hand_pose"].clone().detach().cpu().numpy()
    )
    seq_data_new["rhand"]["params"]["fullpose"] = (
        optimized_params["right_hand_pose"].clone().detach().cpu().numpy()
    )
    seq_data_new["body"]["params"]["global_orient"] = (
        optimized_params["global_orient"].clone().detach().cpu().numpy()
    )
    seq_data_new["body"]["params"]["body_pose"] = (
        optimized_params["body_pose"].clone().detach().cpu().numpy()
    )
    seq_data_new["body"]["params"]["transl"] = (
        optimized_params["transl"].clone().detach().cpu().numpy()
    )
    return seq_data_new


def main():
    """
    Example usage of the retargeting functions
    """
    # base_out_path = os.path.join(PROJECT_DIR, 'DATASETS', 'DATA_GRAB_RETARGETED_FixedFloor')
    base_out_path = os.path.join(TMP_DIR, "debug_retargeting")
    create_new_dir(base_out_path)
    # Default parameters
    num_iterations = 200
    lr = 0.008
    seq_paths = get_all_grab_seq_paths(GRAB_ORIG_DATA_PATH)  # Process first 3 sequences
    seq_paths = np.random.RandomState(42).choice(seq_paths, size=10, replace=False)
    for seq_path in tqdm(seq_paths):
        print(f"Processing {seq_path}...")
        seq_data = parse_npz(seq_path)
        seq_data_new = optimize_neutral_motion(
            seq_data, num_iterations=num_iterations, lr=lr
        )

        rel_path = os.path.relpath(seq_path, GRAB_DATA_PATH)
        new_seq_path = os.path.join(base_out_path, rel_path)
        os.makedirs(os.path.dirname(new_seq_path), exist_ok=True)
        # np.savez(new_seq_path, **seq_data_new)
        np.savez_compressed(new_seq_path, **seq_data_new)


if __name__ == "__main__":
    main()
