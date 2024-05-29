#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging as log
import math
import os
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import utils.logging as logging
from utils.misc import get_class_names

logger = logging.get_logger(__name__)
log.getLogger("matplotlib").setLevel(log.ERROR)


class TensorboardWriter(object):
    """
    Helper class to log information to Tensorboard.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        # class_names: list of class names.
        # cm_subset_classes: a list of class ids -- a user-specified subset.
        # parent_map: dictionary where key is the parent class name and
        #   value is a list of ids of its children classes.
        # hist_subset_classes: a list of class ids -- user-specified to plot histograms.
        (
            self.class_names,
            self.cm_subset_classes,
            self.parent_map,
            self.hist_subset_classes,
        ) = (None, None, None, None)
        self.cfg = cfg
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        if cfg.TENSORBOARD.LOG_DIR == "":
            log_dir = os.path.join(
                cfg.OUTPUT_DIR, "runs-{}".format(cfg.DATA.NAME), time_str
            )
        else:
            log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TENSORBOARD.LOG_DIR)

        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(
            "To see logged results in Tensorboard, please launch using the command \
            `tensorboard  --port=<port-number> --logdir {}`".format(
                log_dir
            )
        )


    def add_scalars(self, data_dict, global_step=None):
        """
        Add multiple scalars to Tensorboard logs.
        Args:
            data_dict (dict): key is a string specifying the tag of value.
            global_step (Optinal[int]): Global step value to record.
        """
        if self.writer is not None:
            for key, item in data_dict.items():
                self.writer.add_scalar(key, item, global_step)



    def add_video(self, vid_tensor, tag="Video Input", global_step=None, fps=4):
        """
        Add input to tensorboard SummaryWriter as a video.
        Args:
            vid_tensor (tensor): shape of (B, T, C, H, W). Values should lie
                [0, 255] for type uint8 or [0, 1] for type float.
            tag (Optional[str]): name of the video.
            global_step(Optional[int]): current step.
            fps (int): frames per second.
        """
        self.writer.add_video(tag, vid_tensor, global_step=global_step, fps=fps)

    def plot_weights_and_activations(
        self,
        weight_activation_dict,
        tag="",
        normalize=False,
        global_step=None,
        batch_idx=None,
        indexing_dict=None,
        heat_map=True,
    ):
        """
        Visualize weights/ activations tensors to Tensorboard.
        Args:
            weight_activation_dict (dict[str, tensor]): a dictionary of the pair {layer_name: tensor},
                where layer_name is a string and tensor is the weights/activations of
                the layer we want to visualize.
            tag (Optional[str]): name of the video.
            normalize (bool): If True, the tensor is normalized. (Default to False)
            global_step(Optional[int]): current step.
            batch_idx (Optional[int]): current batch index to visualize. If None,
                visualize the entire batch.
            indexing_dict (Optional[dict]): a dictionary of the {layer_name: indexing}.
                where indexing is numpy-like fancy indexing.
            heatmap (bool): whether to add heatmap to the weights/ activations.
        """
        for name, array in weight_activation_dict.items():
            if batch_idx is None:
                # Select all items in the batch if batch_idx is not provided.
                batch_idx = list(range(array.shape[0]))
            if indexing_dict is not None:
                fancy_indexing = indexing_dict[name]
                fancy_indexing = (batch_idx,) + fancy_indexing
                array = array[fancy_indexing]
            else:
                array = array[batch_idx]
            add_ndim_array(
                self.writer,
                array,
                tag + name,
                normalize=normalize,
                global_step=global_step,
                heat_map=heat_map,
            )

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()




def add_ndim_array(
    writer,
    array,
    name,
    nrow=None,
    normalize=False,
    global_step=None,
    heat_map=True,
):
    """
    Visualize and add tensors of n-dimentionals to a Tensorboard SummaryWriter. Tensors
    will be visualized as a 2D grid image.
    Args:
        writer (SummaryWriter): Tensorboard SummaryWriter.
        array (tensor): tensor to visualize.
        name (str): name of the tensor.
        nrow (Optional[int]): number of 2D filters in each row in the grid image.
        normalize (bool): whether to normalize when we have multiple 2D filters.
            Default to False.
        global_step (Optional[int]): current step.
        heat_map (bool): whether to add heat map to 2D each 2D filters in array.
    """
    if array is not None and array.ndim != 0:
        if array.ndim == 1:
            reshaped_array = array.unsqueeze(0)
            if nrow is None:
                nrow = int(math.sqrt(reshaped_array.size()[1]))
            reshaped_array = reshaped_array.view(-1, nrow)
            if heat_map:
                reshaped_array = add_heatmap(reshaped_array)
                writer.add_image(
                    name,
                    reshaped_array,
                    global_step=global_step,
                    dataformats="CHW",
                )
            else:
                writer.add_image(
                    name,
                    reshaped_array,
                    global_step=global_step,
                    dataformats="HW",
                )
        elif array.ndim == 2:
            reshaped_array = array
            if heat_map:
                heatmap = add_heatmap(reshaped_array)
                writer.add_image(
                    name, heatmap, global_step=global_step, dataformats="CHW"
                )
            else:
                writer.add_image(
                    name,
                    reshaped_array,
                    global_step=global_step,
                    dataformats="HW",
                )
        else:
            last2_dims = array.size()[-2:]
            reshaped_array = array.view(-1, *last2_dims)
            if heat_map:
                reshaped_array = [
                    add_heatmap(array_2d).unsqueeze(0)
                    for array_2d in reshaped_array
                ]
                reshaped_array = torch.cat(reshaped_array, dim=0)
            else:
                reshaped_array = reshaped_array.unsqueeze(1)
            if nrow is None:
                nrow = int(math.sqrt(reshaped_array.size()[0]))
            img_grid = make_grid(
                reshaped_array, nrow, padding=1, normalize=normalize
            )
            writer.add_image(name, img_grid, global_step=global_step)


def add_heatmap(tensor):
    """
    Add heatmap to 2D tensor.
    Args:
        tensor (tensor): a 2D tensor. Tensor value must be in [0..1] range.
    Returns:
        heatmap (tensor): a 3D tensor. Result of applying heatmap to the 2D tensor.
    """
    assert tensor.ndim == 2, "Only support 2D tensors."
    # Move tensor to cpu if necessary.
    if tensor.device != torch.device("cpu"):
        arr = tensor.cpu()
    else:
        arr = tensor
    arr = arr.numpy()
    # Get the color map by name.
    cm = plt.get_cmap("viridis")
    heatmap = cm(arr)
    heatmap = heatmap[:, :, :3]
    # Convert (H, W, C) to (C, H, W)
    heatmap = torch.Tensor(heatmap).permute(2, 0, 1)
    return heatmap
