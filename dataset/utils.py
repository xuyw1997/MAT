import torch
import numpy as np
import math
from numpy.random import randint
import torchvision
from .transform import * 

def trim_pad_to_fixed_length(sent, length):
    ori_len = len(sent)
    if len(sent) >= length:
        return sent[:length], torch.ones(length)
    else:
        txt_mask = torch.zeros(length)
        txt_mask[:ori_len] = 1
        return sent + ['<pad>' for _ in range(length - len(sent))], txt_mask
    
def label_time_scale(num_frame, duration, times):
    # label scaled to frame-level
    # return **start index rounded**  and **end index rounded(should add 1 when transformed into second)**
    fps = num_frame / duration
    start_frame = int(fps * times[0])
    end_frame = int(fps * times[1])
    if end_frame >= num_frame:
        end_frame = num_frame - 1
    if start_frame > end_frame:
        start_frame = end_frame
    assert start_frame <= end_frame
    assert 0 <= start_frame < num_frame
    assert 0 <= end_frame < num_frame
    label = torch.tensor([start_frame, end_frame])
    return label


def get_sequence(cfg, num_video_frame, clip_index, num_clip):
    num_sample_frame = cfg.DATA.NUM_SAMPLE_FRAME
    sampling_rate = cfg.DATA.SAMPLING_RATE
    clip_length = sampling_rate * num_sample_frame

    # over-long video
    if num_video_frame > num_clip * clip_length:
        start_index = np.linspace(0, num_video_frame-1, num_clip+1)[:-1].astype(np.int64)
        return [start_index[clip_index]+ i for i in range(0, clip_length, sampling_rate)]


    if clip_index != num_clip - 1:
        return [clip_index * clip_length + i for i in range(0, clip_length, sampling_rate)]
    else:
        frame_index = np.linspace((num_clip - 1) * clip_length, num_video_frame, clip_length).astype(np.int64)
        return [frame_index[i] for i in range(0, clip_length, sampling_rate)]


def get_frame_sequence(cfg, num_video_frame, rand_offset=False):
    num_sample_frame = cfg.DATA.NUM_SAMPLE_FRAME

    average_duration = num_video_frame / num_sample_frame
    if average_duration > 0:
        offsets = np.multiply(list(range(num_sample_frame)), average_duration).astype(np.int64)
        if rand_offset:
            offsets +=  randint(max(average_duration, 1), size=num_sample_frame)
    else:
        offsets = np.linspace(0, num_video_frame-1, num_sample_frame).astype(np.int64)
    return offsets
    

def get_transform(input_size, input_mean, input_std, split='train'):
    if split == 'train':
        unique = torchvision.transforms.Compose([
            GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip(),
            # GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            # GroupRandomGrayscale(p=0.2)
        ])
    else:
        unique = torchvision.transforms.Compose([GroupScale(input_size),
                                                 GroupCenterCrop(input_size)])
    # unique = torchvision.transforms.Compose([GroupScale(input_size),
    #                                              GroupCenterCrop(input_size)])
    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        # I3DToTorchFormatTensor(div=True)
        # GroupNormalize(input_mean,input_std)
    ])
    return torchvision.transforms.Compose([unique, common])

def get_clip_gt(times, duration, clip_index, num_clip):

    clip_stamp = np.linspace(0, duration, num_clip+1)
    clip_start = clip_stamp[clip_index]
    clip_end = clip_stamp[clip_index + 1]

    if clip_end <= times[0]:
        is_end_clip = 0
        reg_start = 0
        reg_end = 1
    elif times[0] < clip_end and clip_end <= times[1]:
        is_end_clip = (clip_end - times[0]) / (times[1] - times[0])
        # reg_start = times[0] / clip_end
        reg_start = math.log(clip_end - times[0])
        reg_end = 1
    elif clip_start < times[1] and times[1] < clip_end:
        is_end_clip = 1
        # reg_start = times[0] / times[1]
        reg_start = math.log(times[1] - times[0])
        reg_end = (times[1] - clip_start) / (clip_end - clip_start)
    else:
        is_end_clip = (times[1] - times[0]) / (clip_end - times[0])
        # reg_start = times[0] / clip_start
        reg_start = math.log(clip_start - times[0])
        reg_end = 0

    assert 0 <= is_end_clip and is_end_clip <= 1
    # assert 0 <= reg_start and reg_start <= 1, f'{clip_start}, {clip_end}, {times[0]}, {times[1]}'
    
    assert 0 <= reg_end and reg_end <= 1

    label = torch.tensor([is_end_clip, reg_start, reg_end], dtype=torch.float)
    return label


def cal_gt(gt, total_frame, bsz):
    axis = torch.arange(total_frame, device=gt.device).unsqueeze(0).expand(bsz, -1)
    start_loc_gt = gt[:, 0, None] / (axis + 1)
    start_loc_mask = axis >= gt[:, 0, None]

    i_end = torch.minimum(gt[:, 1, None], axis)
    i = (i_end - gt[:, 0, None] + 1).clamp(min=0)
    u_end = torch.maximum(gt[:, 1, None], axis)
    u = (u_end - gt[:, 0, None] + 1).clamp(min=0)
    iou_gt = i / u

    return iou_gt, start_loc_gt, start_loc_mask