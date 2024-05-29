#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np

import torch

import utils.checkpoint as cu

import utils.logging as logging
from dataset import loader
from model import build_model
from utils.meters import ValMeter
from train_net import eval_epoch3

logger = logging.get_logger(__name__)




def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
  
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.CONFIG_FILE)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    
    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    val_loader = loader.construct_loader(cfg, "test")

    val_meter = ValMeter(len(val_loader), cfg)
    logger.info("Testing model for {} iterations".format(len(val_loader)))



    # Set up writer for logging to Tensorboard format.
    writer = None

    # # Perform multi-view test on the entire dataset.
    eval_epoch3(
        val_loader,
        model,
        val_meter,
        0,
        cfg,
        writer,
        save_preds=True
    )
    if writer is not None:
        writer.close()

    logger.info("testing done")

