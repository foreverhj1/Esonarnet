import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.feature import hog
from skimage import feature as ft
from nets.resnet import resnet50
from nets.vgg import VGG16

from efficientvit.seg_model_zoo import create_seg_model
import cv2

import torch.nn.functional as F
from .attention import BiFusion_block


class Esonarnet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Esonarnet, self).__init__()
        in_filters  = [72, 144, 288, 192]
        out_filters = [32, 64, 128, 256]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0
        self.backbone = efficientvit_backbone_b0()  
        self.final = nn.Conv2d(32, num_classes, 1)
        self.hog_encoder1 = nn.Conv2d(3, 16, 3, 2, padding=1)
        self.hog_encoder2 = nn.Conv2d(16, 32, 3, 2, padding=1)

        drop_rate = 0.2
        self.biFusion =  BiFusion_block(ch_1=64, ch_2=32, r_2=4, ch_int=64, ch_out=64, drop_rate=drop_rate/2)
        
        
    def forward(self, x, hog_image):
        # 
        
        self.cnn_encoder
        self.LGFI
        self.trans_stage
        self.GLAU
        
        hog_feature = self.hog_encoder1(hog_image)
        hog_feature = self.hog_encoder2(hog_feature)
        # s = time.time()

        features = self.backbone(x)
        feat1 = features['stage0']
        feat2 = features['stage1']
        feat3 = features['stage2']
        feat4 = features['stage3']
        feat5 = features['stage4']
        # print('1', time.time() -s)
        # print(feat1.shape, feat2.shape, feat3.shape, feat4.shape, feat5.shape)
        
        # how to fusion: 
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        # up2 = torch.cat([up2, hog_feature], 1)
        # print(hog_feature.shape, up2.shape)
        # s = time.time()
        up2 = self.biFusion(up2, hog_feature)
        # print('2', time.time() -s)
        
        up1 = self.up_concat1(feat1, up2)
        up1 = self.up_conv(up1)
        final = self.final(up1)

        return final

    def freeze_backbone(self):
        return

    def unfreeze_backbone(self):
        return
