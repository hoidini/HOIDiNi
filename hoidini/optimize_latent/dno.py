import math
from typing import Optional
import torch
from tqdm import tqdm
from dataclasses import dataclass, field

from hoidini.optimize_latent.dno_losses import DnoCondition


@dataclass
class DNOOptions:
    num_opt_steps: int = field(
        default=500,
        metadata={
            "help": "Number of optimization steps (300 for editing, 500 for refinement, can go further for better results)"
        },
    )
    lr: float = field(default=5e-2, metadata={"help": "Learning rate"})
    perturb_scale: float = field(
        default=0, metadata={"help": "scale of the noise perturbation"}
    )
    diff_penalty_scale: float = field(
        default=0,
        metadata={
            "help": "penalty for the difference between the final z and the initial z"
        },
    )
    lr_warm_up_steps: int = field(
        default=50, metadata={"help": "Number of warm-up steps for the learning rate"}
    )
    lr_decay_steps: int = field(
        default=None,
        metadata={"help": "Number of decay steps (if None, then set to num_opt_steps)"},
    )
    decorrelate_scale: float = field(
        default=1000, metadata={"help": "penalty for the decorrelation of the noise"}
    )
    decorrelate_dim: int = field(
        default=3,
        metadata={
            "help": "dimension to decorrelate (we usually decorrelate time dimension)"
        },
    )

    loss_stop: float = field(
        default=None, metadata={"help": "Stop optimization if loss is below this value"}
    )

    def __post_init__(self):
        # if lr_decay_steps is not set, then set it to num_opt_steps
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.num_opt_steps


class DNO:
    """
    Args:
        start_z: (N, 263, 1, 120)
    """

    def __init__(
        self,
        model,
        criterion,
        start_z,
        conf: DNOOptions,
    ):
        self.model = model
        self.criterion: DnoCondition = criterion
        # for diff penalty
        self.start_z = start_z.detach()
        self.conf = conf

        self.current_z = self.start_z.clone().requires_grad_(True)
        # excluding the first dimension (batch size)
        self.dims = list(range(1, len(self.start_z.shape)))

        self.optimizer = torch.optim.Adam([self.current_z], lr=conf.lr)

        self.lr_scheduler = []
        if conf.lr_warm_up_steps > 0:
            self.lr_scheduler.append(
                lambda step: warmup_scheduler(step, conf.lr_warm_up_steps)
            )
        self.lr_scheduler.append(
            lambda step: cosine_decay_scheduler(
                step, conf.lr_decay_steps, conf.num_opt_steps, decay_first=False
            )
        )

        self.step_count = 0
        # history of the optimization (for each step and each instance in the batch)
        # hist = {
        #    "step": [step] * batch_size,
        #    "lr": [lr] * batch_size,
        #    ...
        # }
        self.hist = []

    def __call__(
        self, num_steps: Optional[int] = None, plot=False, save_loss_graph=None
    ):
        if num_steps is None:
            num_steps = self.conf.num_opt_steps

        batch_size = self.start_z.shape[0]
        with tqdm(
            range(num_steps), desc="DNO Optimization", dynamic_ncols=True
        ) as prog:
            for i in prog:
                info = {"step": [self.step_count] * batch_size}

                # learning rate scheduler
                lr_frac = 1
                if len(self.lr_scheduler) > 0:
                    for scheduler in self.lr_scheduler:
                        lr_frac *= scheduler(self.step_count)
                    self.set_lr(self.conf.lr * lr_frac)
                info["lr"] = [self.conf.lr * lr_frac] * batch_size

                # criterion
                x = self.model(self.current_z)
                # [batch_size,]
                if isinstance(
                    x, list
                ):  # Roey: the model returns the result of all diffusion steps
                    x = x[-1]
                loss, loss_dict = self.criterion(x)
                for k, v in loss_dict.items():
                    info[k] = v.detach().cpu()  # if v is not None else [0] * batch_size

                assert loss.shape == (batch_size,), loss.shape
                info["loss"] = loss.detach().cpu()
                loss = loss.sum()

                # diff penalty
                if self.conf.diff_penalty_scale > 0:
                    # [batch_size,]
                    loss_diff = (self.current_z - self.start_z).norm(p=2, dim=self.dims)
                    assert loss_diff.shape == (batch_size,)
                    loss += self.conf.diff_penalty_scale * loss_diff.sum()
                    info["loss_diff"] = loss_diff.detach().cpu()
                else:
                    info["loss_diff"] = [0] * batch_size

                # decorrelate
                if self.conf.decorrelate_scale > 0:
                    loss_decorrelate = noise_regularize_1d(
                        self.current_z,
                        dim=self.conf.decorrelate_dim,
                    )
                    assert loss_decorrelate.shape == (batch_size,)
                    loss += self.conf.decorrelate_scale * loss_decorrelate.sum()
                    info["loss_decorrelate"] = loss_decorrelate.detach().cpu()
                else:
                    info["loss_decorrelate"] = [0] * batch_size

                # backward
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # log grad norm (before)
                info["grad_norm"] = (
                    self.current_z.grad.norm(p=2, dim=self.dims).detach().cpu()
                )

                # grad mode
                self.current_z.grad.data /= self.current_z.grad.norm(
                    p=2, dim=self.dims, keepdim=True
                )

                # optimize z
                self.optimizer.step()

                # noise perturbation
                # match the noise fraction to the learning rate fraction
                noise_frac = lr_frac
                info["perturb_scale"] = [
                    self.conf.perturb_scale * noise_frac
                ] * batch_size

                noise = torch.randn_like(self.current_z)
                self.current_z.data += noise * self.conf.perturb_scale * noise_frac

                # log the norm(z - start_z)
                info["diff_norm"] = (
                    (self.current_z - self.start_z)
                    .norm(p=2, dim=self.dims)
                    .detach()
                    .cpu()
                )

                # log current z
                info["z"] = self.current_z.detach().cpu()
                info["x"] = x.detach().cpu()

                self.step_count += 1
                self.hist.append(info)
                # prog.set_postfix({"loss": info["loss"].mean().item()})
                log_dict = {}
                log_dict["loss"] = info["loss"].mean().item()
                for b in range(batch_size):
                    for k, v in info.items():
                        if k in ["step", "x", "z"]:
                            continue
                        log_dict[f"{k}_{b}"] = round(
                            v[b].item() if isinstance(v[b], torch.Tensor) else v[b], 4
                        )
                prog.set_postfix(log_dict)

                # if i % 20 == 0:
                #     print(f"Step {i} loss term values:")
                #     for k, v in log_dict.items():
                #         print(f"{10 * ' '}{k}: {v}")
                # ##### My addition ##########
                if (
                    self.conf.loss_stop is not None
                    and loss.item() < self.conf.loss_stop
                ):
                    break
                # ##### /My addition ##########

            # output is a list (over batch) of dict (over keys) of lists (over steps)
            hist = []
            for i in range(batch_size):
                hist.append({})
                for k in self.hist[0].keys():
                    hist[-1][k] = [info[k][i] for info in self.hist]

            if plot or save_loss_graph:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 3))
                for i in range(batch_size):
                    plt.plot(hist[i]["step"], hist[i]["loss"], label=f"Instance {i}")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title("Loss over Optimization Steps")
                plt.legend()
                if plot:
                    plt.show()
                if save_loss_graph:
                    plt.savefig(save_loss_graph)

            return {
                # last step's z
                "z": self.current_z.detach(),
                # previous steps' x
                "x": x.detach(),
                "hist": hist,  # hist[batch][loss_name][step]
            }

    def set_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr


def warmup_scheduler(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1


def cosine_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using cosine decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (math.cos((step) / decay_steps * math.pi) + 1) / 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return (
            math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1
        ) / 2


def noise_regularize_1d(noise, stop_at=2, dim=3):
    """
    Args:
        noise (torch.Tensor): (N, C, 1, size)
        stop_at (int): stop decorrelating when size is less than or equal to stop_at
        dim (int): the dimension to decorrelate
    """
    all_dims = set(range(len(noise.shape)))
    loss = 0
    size = noise.shape[dim]

    # pad noise in the size dimention so that it is the power of 2
    if size != 2 ** int(math.log2(size)):
        new_size = 2 ** int(math.log2(size) + 1)
        pad = new_size - size
        pad_shape = list(noise.shape)
        pad_shape[dim] = pad
        pad_noise = torch.randn(*pad_shape).to(noise.device)

        noise = torch.cat([noise, pad_noise], dim=dim)
        size = noise.shape[dim]

    while True:
        # this loss penalizes spatially correlated noise
        # the noise is rolled in the size direction and the dot product is taken
        # (bs, )
        loss = loss + (noise * torch.roll(noise, shifts=1, dims=dim)).mean(
            # average over all dimensions except 0 (batch)
            dim=list(all_dims - {0})
        ).pow(2)

        # stop when size is 8
        if size <= stop_at:
            break

        # (N, C, 1, size) -> (N, C, 1, size // 2, 2)
        noise_shape = list(noise.shape)
        noise_shape[dim] = size // 2
        noise_shape.insert(dim + 1, 2)
        noise = noise.reshape(noise_shape)
        # average pool over (2,) window
        noise = noise.mean([dim + 1])
        size //= 2

    return loss
