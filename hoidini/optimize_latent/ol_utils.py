from matplotlib import pyplot as plt
from hoidini.optimize_latent.dno import DNO, DNOOptions
from hoidini.optimize_latent.dno_loss_functions import (
    DnoCondHandsIntersection,
    DnoCondVertsDist,
)
import torch


def get_static_criterion(smpl_fk, normalizer, n_joints, n_simplify_faces=700):
    loss_fn_lst = [
        (0.2, DnoCondVertsDist(smpl_fk, normalizer, n_joints)),
        (
            1,
            DnoCondHandsIntersection(
                smpl_fk, normalizer, n_joints, n_simplify_faces=700
            ),
        ),
    ]

    def criterion(x, **model_kwargs):
        result = 0
        for alpha, loss_fn in loss_fn_lst:
            result += alpha * loss_fn(x, **model_kwargs)
        return result

    return criterion


def plot_lr_scheduler(dno_opts: DNOOptions):
    dno = DNO(model=None, criterion=None, start_z=torch.ones(1), conf=dno_opts)
    lr_scheduler = dno.lr_scheduler
    lrs = []
    steps = []
    for step in range(dno.conf.num_opt_steps):
        lr_frac = 1
        if len(lr_scheduler) > 0:
            for scheduler in lr_scheduler:
                lr_frac *= scheduler(step)
            lr = dno.conf.lr * lr_frac

        lrs.append(lr)
        steps.append(step)
    plt.figure()
    plt.plot(steps, lrs)
    plt.show()
