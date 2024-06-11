'''
Author: orchxuhu
Date: 2024-06-10 18:34:52
LastEditors: orchxuhu
LastEditTime: 2024-06-11 16:32:51
Description: 

Copyright ORCA 2024, All Rights Reserved. 
'''
# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import os
from inspect import signature
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


def is_parallel(model: nn.Module) -> bool:
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device


def get_same_padding(
        kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: any or None = None,
    scale_factor=None,
    mode: str = "bicubic",
    align_corners: bool or None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x,
                             size=size,
                             scale_factor=scale_factor,
                             mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def build_kwargs_from_config(config: dict, target_func):
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def load_state_dict_from_file(file, only_state_dict=True):
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


class LGFI_block(nn.Module):
    def __init__(self):
        super(LGFI_block, self).__init__()
        self.conv1 = Conv(128, 32, 3, bn=True, relu=False)
        self.conv2 = Conv(128, 32, 5, bn=True, relu=False)
        self.conv3 = Conv(128, 32, 7, bn=True, relu=False)
        self.conv4 = Conv(128, 32, 9, bn=True, relu=False)
        self.conv5 = Conv(1, 1, 1, bn=True, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.cat(
            (self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)),
            dim=1)
        x1 = x * x1
        x2 = torch.max(x, dim=1, keepdim=True)
        x2 = x * self.sigmoid(self.conv5(x2))
        return x1 + x2


class GLAU_block(nn.Module):
    def __init__(self):
        super(GLAU_block, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = Conv(128, 128, 1, bn=True, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t, c):
        t1 = self.conv1(self.global_avg_pool(t))
        t1 = self.sigmoid(t1)
        return t * t1 * c
