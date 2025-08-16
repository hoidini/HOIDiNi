import math
import os
import torch
from copy import deepcopy
from typing import List, Optional

from hoidini.closd.diffusion_planner.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
)
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.cphoi.cphoi_dataset import TfmsManager
from hoidini.datasets.grab.grab_utils import grab_seq_path_to_unique_name
from hoidini.closd.diffusion_planner.utils.model_util import (
    create_model_and_diffusion,
    load_saved_model,
)
from hoidini.closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
from hoidini.object_contact_prediction.cpdm_dataset import (
    ContactPairsSequence,
    CpsMetadata,
    load_contact_pairs_sequence,
)
from hoidini.optimize_latent.dno import DNO
from hoidini.datasets.grab.grab_object_records import ContactRecord
from hoidini.general_utils import TMP_DIR, create_new_dir
from hoidini.optimize_latent.dno_losses import DnoCondition
from hoidini.resource_paths import GRAB_DATA_PATH


def convert_optstep_to_batch(optstep_list):
    bs = next(iter(optstep_list[0].values())).shape[0]
    loss_names = optstep_list[0].keys()
    result = [dict() for _ in range(bs)]
    for loss_name in loss_names:
        for b in range(bs):
            result[b][loss_name] = [step[loss_name][b].item() for step in optstep_list]
    return result


class ClassifierGuidanceCondFn:
    def __init__(self, dno_cond, return_grad: bool = False):
        self._loss_history = []
        self.return_grad = return_grad
        self.dno_cond = dno_cond

    def __call__(self, hat_x, t, p_mean_var, **kwargs):
        loss, losses = self.dno_cond(hat_x)
        losses = {k: v.clone().detach().cpu() for k, v in losses.items()}
        self._loss_history.append(losses)
        if len(self._loss_history) % 50 == 0:
            print(f"Loss history: {self._loss_history[-1]}")
        loss = loss.mean(dim=0)  # (bs, ) -> (,)
        if self.return_grad:
            return torch.autograd.grad(loss, hat_x)[0]
        return loss


def get_solver(
    sample_fn: GaussianDiffusion.ddim_sample_loop,
    model,
    shape,
    cur_model_kwargs,
    enable_grad: bool = False,
):
    def solver(z):
        return sample_fn(
            model,
            shape=shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=cur_model_kwargs,
            skip_timesteps=0,
            progress=False,
            enable_grad=enable_grad,
        )

    return solver


def load_model(args, n_feats):
    model, diffusion = create_model_and_diffusion(args, n_feats)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    if args.gen_guidance_param != 1:
        model = ClassifierFreeSampleModel(model, args.cfg_type)
    model.to(dist_util.dev())
    model.eval()
    return model, diffusion


def save_real_contact_pairs_seqs():
    def _save_real_contact_pairs(grab_seq_path: str, contact_pairs_seq_path: str):
        contact_pairs, object_name = load_contact_pairs_sequence(grab_seq_path)
        contact_pairs.metadata = CpsMetadata(
            grab_seq_path=grab_seq_path,
            object_name=object_name,
            text="",
        )
        contact_pairs.save(contact_pairs_seq_path)
        return contact_pairs

    grab_seq_paths = [
        os.path.join(GRAB_DATA_PATH, "s3/gamecontroller_play_1.npz"),
        os.path.join(GRAB_DATA_PATH, "s4/bowl_drink_1.npz"),
        # os.path.join(GRAB_DATA_PATH, "s4/hammer_use_1.npz"),
        # os.path.join(GRAB_DATA_PATH, "s7/binoculars_see_1.npz"),
        # os.path.join(GRAB_DATA_PATH, "s4/spherelarge_lift.npz"),
    ]
    out_dir = os.path.join(TMP_DIR, "real_contact_pairs")
    print(f"\nSaving real contact pairs to\n {out_dir}\n")
    create_new_dir(out_dir)
    for grab_seq_path in grab_seq_paths:
        out_contact_pairs_path = os.path.join(
            out_dir,
            f"real_contact_pairs_{grab_seq_path_to_unique_name(grab_seq_path)}.npz",
        )
        print(f"Saving real contact pairs to {out_contact_pairs_path}")
        _save_real_contact_pairs(grab_seq_path, out_contact_pairs_path)


def auto_regressive_dno(
    args,
    model,
    required_frames,
    sample_fn,
    noise_opt_conf,
    shape,
    model_kwargs,
    use_dno,
    dno_cond: Optional[DnoCondition] = None,
    seed=None,
    tfms_manager: Optional[TfmsManager] = None,
    sequence_conditions: Optional[List[ContactRecord | ContactPairsSequence]] = None,
    use_classifier_guidance: bool = False,
    diffusion: Optional[GaussianDiffusion] = None,
) -> tuple[torch.Tensor, list[list[dict[str, list[float]]]]]:
    """
    sequence_conditions are assumed to include the context frames!

    """
    # assert sum([use_dno, use_classifier_guidance]) < 2
    assert shape[-1] == required_frames
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    is_include_prefix = args.autoregressive_include_prefix
    assert is_include_prefix, "No support for non-prefix yet"
    context_len = args.train.context_len
    pred_len = args.train.pred_len

    n_new_frames = (
        required_frames - context_len if is_include_prefix else required_frames
    )
    n_iterations = math.ceil(n_new_frames / pred_len)

    cur_prefix = deepcopy(model_kwargs["y"]["prefix"])
    assert cur_prefix.shape[3] == context_len

    samples_buf_lst = [cur_prefix]
    full_batch = torch.cat(samples_buf_lst, dim=-1)

    autoregressive_shape = list(deepcopy(shape))
    autoregressive_shape[-1] = pred_len

    print(
        f"No. of Auto-Regressive Iterations: {n_iterations} (required_frames: {required_frames}, pred_len: {pred_len})"
    )
    optim_info_lst = [] if use_dno else None
    mat_lst = [] if tfms_manager is not None else None
    for ar_step in range(n_iterations):
        cur_model_kwargs = deepcopy(model_kwargs)
        cur_model_kwargs["y"]["prefix"] = cur_prefix
        if tfms_manager is not None:
            tfms_root_global = tfms_manager.get_global_tfms_from_features(full_batch)
            cur_model_kwargs["y"]["tfms_root_global"] = (
                tfms_root_global[:, -context_len].clone().detach()
            )
            mat_lst.append(cur_model_kwargs["y"]["tfms_root_global"])

        z = torch.randn(autoregressive_shape, generator=generator).to(dist_util.dev())

        solver = get_solver(
            sample_fn,
            model,
            autoregressive_shape,
            cur_model_kwargs,
            enable_grad=use_dno and not use_classifier_guidance,
        )

        if use_dno:
            start = ar_step * pred_len + context_len
            end = start + pred_len
            dno_cond.reset()
            dno_cond.set_global_prefix_lst(samples_buf_lst)
            if sequence_conditions is not None:
                current_seqs = []
                for seq in sequence_conditions:
                    seq = seq.cut(start, end)
                    if len(seq) < pred_len:
                        seq = seq.extend(pred_len)
                    current_seqs.append(seq)
                dno_cond.set_current_seq(current_seqs)
            if not use_classifier_guidance:
                ####################
                # DNO
                ####################
                dno = DNO(
                    model=solver, criterion=dno_cond, start_z=z, conf=noise_opt_conf
                )
                dno_result = dno()
                sample = dno_result["x"]
                loss_dict = dno_result["hist"]
                optim_info_lst.append(loss_dict)
            elif use_classifier_guidance:
                ####################
                # Classifier Guidance (for comparison with DNO)
                ####################
                cond_fn = ClassifierGuidanceCondFn(dno_cond, return_grad=True)
                sample_fn: GaussianDiffusion.p_sample_loop = diffusion.p_sample_loop
                sample = sample_fn(
                    model,
                    shape=shape,
                    noise=z,
                    clip_denoised=False,
                    model_kwargs=cur_model_kwargs,
                    skip_timesteps=0,
                    progress=True,
                    cond_fn=cond_fn,
                    return_with_prefix=False,
                    # recon_guidance=True,
                    cond_fn_with_grad=True,
                )

                _loss_history = cond_fn._loss_history
                loss_dict = convert_optstep_to_batch(_loss_history)
                optim_info_lst.append(loss_dict)
                sample = sample.detach()

        else:
            sample = solver(z)

        assert sample.shape[3] == pred_len
        sample = sample.clone().detach()
        samples_buf_lst.append(sample)
        cur_prefix = sample[..., -context_len:].clone().detach()
        full_batch = (
            torch.cat(samples_buf_lst, dim=-1).clone().detach()
        )  # [..., :required_frames]

    if not is_include_prefix:
        full_batch = full_batch[..., context_len:]
    full_batch = full_batch[..., :required_frames].clone().detach()
    if optim_info_lst is not None:
        optim_info_lst = list(
            map(list, zip(*optim_info_lst))
        )  # (ar_step, batch_size) -> (batch_size, ar_step)
    if mat_lst is not None:
        mat_lst = list(
            torch.stack(mat_lst, dim=0).permute(1, 0, 2, 3)
        )  # ar*(bs, 4, 4) --> bs*(ar, 4, 4)

    debug_data = {"mat_lst": mat_lst, "optim_info_lst": optim_info_lst}

    assert full_batch.shape[-1] == required_frames
    return full_batch, debug_data


if __name__ == "__main__":
    save_real_contact_pairs_seqs()
