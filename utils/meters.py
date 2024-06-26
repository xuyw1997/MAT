#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
import time
import simplejson
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from utils.env import make_dir


logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret




class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        clip_check = self.clip_count == self.num_clips
        if not all(clip_check):
            logger.warning(
                "clip count Ids={} = {} (should be {})".format(
                    np.argwhere(~clip_check),
                    self.clip_count[~clip_check],
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            mean_ap = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            map_str = "{:.{prec}f}".format(mean_ap * 100.0, prec=2)
            self.stats["map"] = map_str
            self.stats["top1_acc"] = map_str
            self.stats["top5_acc"] = map_str
        else:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                # self.stats["top{}_acc".format(k)] = topk.cpu().numpy()
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque) if len(self.deque) > 0 else 0

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None

        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None

        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self,loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size



    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        avg_loss = self.loss_total / self.num_samples

        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            "loss": avg_loss
        }

        logging.log_json_stats(stats, self.output_dir)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_samples = 0
        self.num_loss_samples = 0
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0

        self.all_ious = []
        self.all_preds = []
        self.output_dir = cfg.OUTPUT_DIR
        self.r_accumulate = [0] * len(cfg.TEST.IOU)
        self.foreground_hit = []


    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.data_timer.reset()
        self.net_timer.reset()

        self.all_ious.clear()
        self.all_preds.clear()
        self.num_samples = 0
        self.r_accumulate = [0] * len(self.r_accumulate)

        self.loss.reset()
        self.loss_total = 0.0
        self.num_loss_samples = 0
        self.foreground_hit.clear()



    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()



    def update_stats7(self, loss, locs, confs, centerness, clip_levels, center, times, duration, clip_mask):
        bsz, T = confs.shape
        
        self.loss.add_value(loss)
        self.num_loss_samples += bsz 
        self.loss_total += loss * bsz
        centerness = torch.softmax(centerness, dim=-1)
        confs = torch.softmax(confs, dim=-1)
        clip_levels = torch.softmax(clip_levels, dim=-1)
        final_conf = confs +  centerness + clip_levels
        
        
        pred_center_idx = final_conf.argmax(dim=1)

        pred_start = center[None, :] - locs[..., 0]
        pred_end = center[None, :] + locs[..., 1]
        pred_start.clamp_(min=0)
        pred_end.clamp_(max=T)
        pred_start = pred_start[range(bsz), pred_center_idx]
        pred_end = pred_end[range(bsz), pred_center_idx]
        pred = torch.stack([pred_start, pred_end], dim=-1).cpu()
        pred= pred / T * duration[:, None]
        clip_mask = clip_mask.cpu()
        center_sec = center[None, :].expand(bsz, -1).cpu() / T * duration[:, None]
        center_sec = center_sec[range(bsz),pred_center_idx.cpu()]
        fg_hit= (center_sec >= times[:, 0]) & (center_sec <= times[:, 1]).cpu()
        fg_hit = fg_hit.masked_select(clip_mask)
        self.foreground_hit.extend(fg_hit.tolist())
        ious = metrics.cal_iou_second(pred, times)
        ious = ious.masked_select(clip_mask)
        for idx, val in enumerate(self._cfg.TEST.IOU):
            self.r_accumulate[idx] += (ious > val).sum().item()
        self.all_ious.extend(ious.tolist())
        
        

   



    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "loss": self.loss.get_win_median(),
        }

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        avg_loss = self.loss_total / self.num_loss_samples
        num_samples = len(self.all_ious)
        assert num_samples == len(self.foreground_hit)
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            "loss": avg_loss,
            "num_samples": num_samples
        }

        
        for idx, val in enumerate(self._cfg.TEST.IOU):
            stats[f'R@1, IoU={val}'] = self.r_accumulate[idx] / num_samples
        stats['mIoU'] = sum(self.all_ious) / num_samples
        stats['fg_hit'] = sum(self.foreground_hit) / len(self.foreground_hit)
        logging.log_json_stats(stats)

    def save_preds(self):
        results_dir = os.path.join(self.output_dir, 'results')
        make_dir(results_dir)
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        filename = os.path.join(results_dir, f"{time_str}.json")
        with open(filename, "w") as outfile:
            simplejson.dump(self.all_preds, outfile)




def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
