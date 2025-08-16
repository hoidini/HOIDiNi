import pickle
import numpy as np
from matplotlib import pyplot as plt


def plot_dno_loss0(optim_history, save_path=None, show=True, keys=None, log_scale=True):
    n_batches = len(optim_history)  # Number of columns
    n_ar_steps = len(optim_history[0])  # Number of rows

    # Create a grid of subplots:
    fig, axs = plt.subplots(
        n_batches,
        n_ar_steps,
        figsize=(n_ar_steps * 5, n_batches * 4),
        sharex=True,
        sharey=True,
    )

    # If there is only one row or one column, ensure axs is 2D (list of lists)
    if n_batches == 1:
        axs = [axs]
    if n_ar_steps == 1:
        axs = [[ax] for ax in axs]

    # Dictionary to store lines for legend
    lines_dict = {}

    # Loop through each batch (row) and each ar_step (column)
    for batch_idx in range(n_batches):
        for ar_idx in range(n_ar_steps):
            ax = axs[batch_idx][ar_idx]
            # Access the data for this particular ar_step and batch.
            batch_data = optim_history[batch_idx][ar_idx]

            # Plot every loss curve on the same subplot.
            for loss_name, loss_values in batch_data.items():
                if loss_name == "smpldata":
                    continue
                if keys is not None and loss_name not in keys:
                    continue
                if loss_name in ["step", "x", "z"]:
                    continue
                # Convert to numpy array if not already
                loss_values = np.array(loss_values)
                # Take mean across all dimensions except the first one
                while loss_values.ndim > 1:
                    loss_values = loss_values.mean(axis=-1)
                # Plot the reduced loss values and store line if not already stored
                line = ax.plot(loss_values)[0]
                if loss_name not in lines_dict:
                    lines_dict[loss_name] = line

            # Set the title for this subplot
            ax.set_title(f"ar_step {ar_idx} | batch {batch_idx}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            if log_scale:
                ax.set_yscale("log")

    # Create a single legend outside all subplots
    fig.legend(
        lines_dict.values(),
        lines_dict.keys(),
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
    )

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def plot_dno_loss(optim_history, save_path=None):
    """
    optim_history[ar_step][loss_name][frame]
    """
    loss_keys = [
        k
        for k in optim_history[0].keys()
        if k not in {"step", "x", "z", "loss_diff", "diff_norm"}
    ]

    n_cols = len(optim_history)
    n_rows = len(loss_keys)

    _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axs = axs.reshape(n_rows, n_cols)
    for loss_ind, loss_name in enumerate(loss_keys):
        if loss_name == "smpldata":
            continue
        for ar_step in range(n_cols):
            ax = axs[loss_ind][ar_step]
            if ar_step == 0:
                ax.set_title(f"{loss_name}")

            ax.plot(optim_history[ar_step][loss_name])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def main():
    pickle_path = "/home/dcor/roeyron/tmp/results_hoi_04_15_00_40/hoi_animation_0__hammer__The_person_is_useing_a_hammer.blend"
    pickle_path = pickle_path.replace(".blend", ".pickle")

    with open(pickle_path, "rb") as f:
        optim_history = pickle.load(f)["optim_info_lst"]

    plot_dno_loss(
        optim_history, save_path=pickle_path.replace(".pickle", "_loss.png"), show=False
    )


if __name__ == "__main__":
    main()
