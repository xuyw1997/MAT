#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]



def cal_iou(predict, gt):
    """

    :param predict: (B, 2)
    :param gt_times: (B, 2)
    :return:
    """
    i_start = torch.maximum(predict[:, 0], gt[:, 0])
    i_end = torch.minimum(predict[:, 1], gt[:, 1])
    i = (i_end - i_start + 1).clamp(min=0)
    u_start = torch.minimum(predict[:, 0], gt[:, 0])
    u_end = torch.maximum(predict[:, 1], gt[:, 1])
    u = (u_end - u_start + 1).clamp(min=0)
    iou = i / u
    return iou

def cal_iou_second(predict, times):
    """

    :param predict: (B, 2)
    :param gt_times: (B, 2)
    :return:
    """
    # predict[:, 1] += 1

    i_start = torch.maximum(predict[:, 0], times[:, 0])
    i_end = torch.minimum(predict[:, 1], times[:, 1])
    i = (i_end - i_start).clamp(min=0)
    u_start = torch.minimum(predict[:, 0], times[:, 0])
    u_end = torch.maximum(predict[:, 1], times[:, 1])
    u = (u_end - u_start).clamp(min=0)
    iou = i / u
    return iou

def cal_iou_second_batch(predict, times):
    """

    :param predict: (*, 2)
    :param gt_times: (*, 2)
    """


    i_start = torch.maximum(predict[..., 0], times[..., 0])
    i_end = torch.minimum(predict[..., 1], times[..., 1])
    i = (i_end - i_start).clamp(min=0)
    u_start = torch.minimum(predict[..., 0], times[..., 0])
    u_end = torch.maximum(predict[..., 1], times[..., 1])
    u = (u_end - u_start).clamp(min=0.00001)
    iou = i / u
    return iou

def cal_iou_single(predict, times):
    i_start = max(predict[0], times[0])
    i_end = min(predict[1], times[1])
    i = (i_end - i_start).clamp(min=0)
    u_start = min(predict[0], times[0])
    u_end = max(predict[1], times[1])
    u = (u_end - u_start).clamp(min=0)
    iou = i / u
    return iou