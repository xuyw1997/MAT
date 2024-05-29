#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(
            inputs, targets
        )
        return loss





class AnchorFreeLoss(nn.Module):
    def __init__(self, cfg):
        super(AnchorFreeLoss, self).__init__()
        self.cfg = cfg
        self.num_sample_frame = cfg.DATA.NUM_SAMPLE_FRAME
        self.num_clip_feature = cfg.DATA.NUM_SAMPLE_FRAME  
        self.center = torch.arange(self.num_clip_feature) + 0.5
        if cfg.DDP.NUM_GPUS > 0:
            self.center = self.center.cuda()
        
        self.window_size = cfg.DATA.WINDOW_SIZE
        if self.num_clip_feature != self.num_sample_frame:
            self.window_size = int((self.num_clip_feature / self.num_sample_frame) * self.window_size)
        
        self.lambdas = cfg.LOSS.LAMBDAS
        if len(self.lambdas) == 0:
            self.lambdas = [1.0] * 4
        self.clip_gt_thres = cfg.LOSS.CLIP_GT_THRES

   
    def forward(self, loc, conf, centerness, clip_level_pred, label,  clip_offset):
        """

        :param loc : B, T, 2   
        :param conf: B, T, 1
        :param clip_level_pred: B, 1
        :param label: B, 2
        :param center: B, T
        :param loc_mask: B, T
        :return:
        """
        
        bsz, T = loc.size(0), loc.size(1)
        if self.num_clip_feature != self.num_sample_frame:
            clip_offset = int((self.num_clip_feature / self.num_sample_frame) * clip_offset)
        center = self.center[clip_offset: clip_offset + self.window_size]
        center = center[None,:].expand(bsz, -1)
       

        label = label * self.num_clip_feature
        conf_gt = (center <= label[:,1, None])  & (center >= label[:, 0, None])
        

        pred_start = center - loc[..., 0]
        pred_start.clamp_(min=0)
        
        
        pred_end = center + loc[..., 1]
        pred_end.clamp_(max=self.num_sample_frame)
  

        start_gt = label[:, None,0].expand(-1, T)
 
        end_gt = label[:, None,1].expand(-1, T)

        left_len = center - start_gt
        right_len = end_gt - center 
        centerness_gt = torch.minimum(left_len, right_len) / torch.maximum(left_len, right_len)
        centerness_gt[centerness_gt < 0] = 0

        i = torch.minimum(pred_end, end_gt) - torch.maximum(pred_start, start_gt)
        i.clamp_(min=0.0)
        u = torch.maximum(pred_end, end_gt) - torch.minimum(pred_start, start_gt)
        iou = i / u.clamp_(min=0.001)
        fg_iou = iou[conf_gt]
        N = max(conf_gt.sum(), 1)
        if conf_gt.numel() > 0:
            iou_loss = (1 - fg_iou).sum()

            loc_loss =  iou_loss / N
        else:
            loc_loss = (pred_start.sum() + pred_end.sum()) /N
        
     
        conf_loss = F.binary_cross_entropy_with_logits(conf, centerness_gt)
        
   
        centerness_loss = F.binary_cross_entropy_with_logits(centerness[conf_gt], fg_iou.detach(), reduction='sum') / N

      
        clip_level_gt = conf_gt.sum(dim=1, keepdim=True) / T 
        
        clip_level_loss = sigmoid_focal_loss(clip_level_pred, clip_level_gt.float(), reduction='mean') 

        
        return  self.lambdas[0] * loc_loss   +  self.lambdas[1] * conf_loss  + self.lambdas[2] * centerness_loss + self.lambdas[3] * clip_level_loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "contrastive_loss": ContrastiveLoss,
    "anchor_free_loss": AnchorFreeLoss
}



def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
