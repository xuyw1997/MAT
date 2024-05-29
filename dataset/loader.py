#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
from functools import partial
from typing import List
import torch
from torch.utils.data._utils.collate import default_collate
from dataset.sampler import MyDistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from . import utils as utils
from .build import build_dataset


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    time = [item for sublist in time for item in sublist]

    inputs, labels, video_idx, time, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(time),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, time, extra_data
    else:
        return inputs, labels, video_idx, time, extra_data


def _collate(batch):
    """
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    frames, total_word_vectors, total_txt_mask, total_label, is_new_clip, clip_mask, msic = zip(*batch)
    frames = default_collate(frames)
    total_word_vectors = default_collate(total_word_vectors)
    total_txt_mask = default_collate(total_txt_mask)
    total_label = default_collate(total_label)
    is_new_clip = default_collate(is_new_clip)
    clip_mask = default_collate(clip_mask)


    extra_data = {}
    for key in msic[0].keys():
        data = [d[key] for d in msic]
        if key == 'sentences':
            extra_data[key] = data
        else:
            extra_data[key] = default_collate(data)

    return frames, total_word_vectors, total_txt_mask, total_label, is_new_clip, clip_mask, extra_data

def tmp_collate(batch):
    frames, word_vectors, txt_mask, word_label, word_mask, label, times, duration, clip_mask = zip(*batch)
    frames, label, times, duration, clip_mask = (
        default_collate(frames),
        default_collate(label),
        default_collate(times),
        default_collate(duration),
        default_collate(clip_mask)
    )
    word_vectors = torch.nn.utils.rnn.pad_sequence(word_vectors, batch_first=True)
    txt_mask = torch.nn.utils.rnn.pad_sequence(txt_mask, batch_first=True)
    word_label = torch.nn.utils.rnn.pad_sequence(word_label, batch_first=True)
    word_mask = torch.nn.utils.rnn.pad_sequence(word_mask, batch_first=True)
    
    return frames, word_vectors, txt_mask, word_label, word_mask, label, times, duration, clip_mask

def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    dataset_name = cfg.DATA.NAME
    if split in ["train"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / 1)
        shuffle = False
        drop_last = True
    elif split in ["val"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / 1)
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        batch_size = int(cfg.TEST.BATCH_SIZE / 1)
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
   
    # Create a sampler for multi-process training
    # sampler = MyDistributedSampler(dataset, total_bsz, num_clip_per_video, shuffle=split == 'train') if cfg.DDP.NUM_GPUS > 1 else None
    sampler = None
    # Create a loader
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        sampler=sampler,
        shuffle=(False if sampler else shuffle),
        collate_fn=tmp_collate
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler 

    if hasattr(loader.dataset, 'shuffle'):
        loader.dataset.shuffle()
