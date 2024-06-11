import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.feature import hog
from skimage import feature as ft
import torch.nn.functional as F

from .nn import (ConvLayer, DSConv, IdentityLayer, SMBConv, OpSequential,
                 ResidualBlock, EfficientViTBlock)
from nets.utils import Upconv, LHDGM_block, LGFI_block, GLAU_block


class EsonarnetBackbone(nn.Module):
    def __init__(
        self,
        width_list,
        depth_list,
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)
        self.lgfi = LGFI_block()
        self.glau = GLAU_block()
        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block,
                                      IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages.append(OpSequential(self.lgfi))
        self.width_list.append(in_channels)
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    ))
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages.append(OpSequential(self.glau))
        self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = SMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor):
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


class Esonarnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Esonarnet, self).__init__()
        self.cnn_trans_stage = create_cnn_tran_brackbone()
        in_filters = [72, 144, 288, 192]
        out_filters = [32, 64, 128, 256]
        self.up_concat2 = Upconv(in_filters[1], out_filters[1])
        self.up_concat1 = Upconv(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3,
                      padding=1),
            nn.ReLU(),
        )
        self.hog_encoder1 = SMBConv(3, 16, 3, 2, 1)
        self.hog_encoder2 = SMBConv(16, 32, 3, 2, 1)
        self.smbconv1 = SMBConv(256, 256, 3, 2, 1)
        self.smbconv2 = SMBConv(256, 128, 3, 2, 1)
        self.smbconv3 = SMBConv(128, 64, 3, 2, 1)
        self.smbconv4 = SMBConv(64, 64, 3, 2, 1)
        self.lhdgm_fusion = LHDGM_block(ch_1=64,
                                        ch_2=32,
                                        r_2=4,
                                        ch_int=64,
                                        ch_out=64,
                                        drop_rate=0.1)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, inputs, hog_image):
        features = self.cnn_trans_stage(inputs)
        feat1 = features['stage0']
        feat2 = features['stage1']
        feat3 = features['stage2']
        hog_feature = self.hog_encoder1(hog_image)
        hog_feature = self.hog_encoder2(hog_feature)
        up2 = self.up_concat2(feat2, feat3)
        up2 = self.smbconv1(up2)
        up2 = self.smbconv2(up2)
        up1 = self.up_concat1(feat2, feat3)
        up1 = self.smbconv3(up1)
        up1 = self.smbconv4(up1)
        up1 = self.lhdgm_fusion(up1, hog_feature)
        final = self.final(up1)
        return final


def create_cnn_tran_brackbone(**kwargs):
    backbone = EsonarnetBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EsonarnetBackbone),
    )
    return backbone


def load_pretrained(name: str,
                    dataset: str,
                    pretrained=True,
                    weight_url: str or None = None,
                    **kwargs):
    model = EsonarnetBackbone()(dataset=dataset, **kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = load_state_dict_from_file(
        'seg/efficientvit/assets/checkpoints/seg/cityscapes/b0.pt')
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    return model