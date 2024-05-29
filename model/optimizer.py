#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

from tokenize import group
import torch
from utils import logging
import utils.lr_policy as lr_policy

logger = logging.get_logger(__name__)

def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    
    groups = {gname : [] for gname in  cfg.SOLVER.PARAM_GROUP_PREFIX}
    groups['rest'] = []
    groups['no_grad'] = []
    g2id = {gname : i for i,gname in  enumerate(cfg.SOLVER.PARAM_GROUP_PREFIX)}
    g2id['rest'] = len(cfg.SOLVER.PARAM_GROUP_PREFIX)
    cfg.SOLVER.PARAM_GROUP_LR.append(1.0)
    named_modules = model.module.named_modules() if isinstance(model, torch.nn.DataParallel) else model.named_modules()
    for name, m in named_modules:
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                groups['no_grad'].append(p)
                continue
            is_rest = True
            for prefix in cfg.SOLVER.PARAM_GROUP_PREFIX:
                if name.startswith(prefix):
                    groups[prefix].append(p)
                    is_rest = False
                    break
            if is_rest:
                groups["rest"].append(p)
            
    optim_params = [
        {   
            "gname": gname, 
            "params": gparam,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "lr_mul": cfg.SOLVER.PARAM_GROUP_LR[g2id[gname]]
        }
        for gname, gparam in groups.items() if gname != "no_grad" and len(gparam) > 0
    ]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == sum([len(x["params"]) for x in optim_params]) + len(groups["no_grad"])
    for x in optim_params:
        logger.info(f'gname={x["gname"]}, num={len(x["params"])} , lr={x["lr_mul"]}')

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        optimizer = torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )
    if cfg.SOLVER.LARS_ON:
        optimizer = LARS(
            optimizer=optimizer, trust_coefficient=0.001, clip=False
        )
    return optimizer


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr * param_group["lr_mul"]


class LARS(object):
    """
    this class is adapted from https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py to
     include ignoring LARS application specific parameters (e.g. 1D params)

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARS. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(
        self,
        optimizer,
        trust_coefficient=0.02,
        clip=True,
        eps=1e-8,
        ignore_1d_param=True,
    ):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip
        self.ignore_1d_param = ignore_1d_param

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = (
                    group["weight_decay"] if "weight_decay" in group else 0
                )
                weight_decays.append(weight_decay)
                apply_LARS = (
                    group["apply_LARS"] if "apply_LARS" in group else True
                )
                if not apply_LARS:
                    continue
                group["weight_decay"] = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.ignore_1d_param and p.ndim == 1:  # ignore bias
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = (
                            self.trust_coefficient
                            * (param_norm)
                            / (grad_norm + param_norm * weight_decay + self.eps)
                        )

                        # clip learning rate for LARS
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group["lr"], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]
