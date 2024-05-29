#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

from utils.parser import load_config, parse_args
import torch
from  utils.multiprocessing import run
from test_net import test
from train_net import train


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    # if cfg.DDP.NUM_GPUS > 1:
    #     torch.multiprocessing.spawn(
    #         run,
    #         nprocs=cfg.DDP.NUM_GPUS,
    #         args=(
    #             cfg.DDP.NUM_GPUS,
    #             func,
    #             init_method,
    #             cfg.SHARD_ID,
    #             cfg.NUM_SHARDS,
    #             cfg.DDP.DIST_BACKEND,
    #             cfg,
    #         ),
    #         daemon=daemon,
    #     )
    # else:
    #     func(cfg=cfg)
    func(cfg=cfg)

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        if args.comment:
            cfg.COMMENT = args.comment
        # Perform training.
        if args.train:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)
        if args.test:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
