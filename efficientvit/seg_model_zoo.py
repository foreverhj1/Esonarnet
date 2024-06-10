# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.models.efficientvit import (
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
    efficientvit_seg_l1,
    efficientvit_seg_l2,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_seg_model"]

import torch
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

REGISTERED_SEG_MODEL = {
    "cityscapes": {
        "b0": "assets/checkpoints/seg/cityscapes/b0.pt",
        "b1": "assets/checkpoints/seg/cityscapes/b1.pt",
        "b2": "assets/checkpoints/seg/cityscapes/b2.pt",
        "b3": "assets/checkpoints/seg/cityscapes/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/cityscapes/l1.pt",
        "l2": "assets/checkpoints/seg/cityscapes/l2.pt",
    },
    "ade20k": {
        "b1": "assets/checkpoints/seg/ade20k/b1.pt",
        "b2": "assets/checkpoints/seg/ade20k/b2.pt",
        "b3": "assets/checkpoints/seg/ade20k/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/ade20k/l1.pt",
        "l2": "assets/checkpoints/seg/ade20k/l2.pt",
    },
}


def create_seg_model(
    name: str, dataset: str, pretrained=True, weight_url: str or None = None, **kwargs
) -> EfficientViTSeg:
    model_dict = {
        "b0": efficientvit_seg_b0,
        "b1": efficientvit_seg_b1,
        "b2": efficientvit_seg_b2,
        "b3": efficientvit_seg_b3,
        #########################
        "l1": efficientvit_seg_l1,
        "l2": efficientvit_seg_l2,
    }

    model_id = "b0"
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)

    if model_id in ["l1", "l2"]:
        set_norm_eps(model, 1e-7)
    
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict      = model.state_dict()
    pretrained_dict = load_state_dict_from_file('seg/efficientvit/assets/checkpoints/seg/cityscapes/b0.pt')
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
