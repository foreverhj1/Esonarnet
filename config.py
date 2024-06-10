import colorsys
import copy
import os.path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.model import Esonarnet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

from torchsummary import summary

class Model(object):
    _defaults = {
        #-------------------------------------------------------------------#
        'model_path': 'model_data.pth',-#
        "num_classes"   : 12,
        "input_shape"   : [512,512],
        "mix_type"      : 1,
        "cuda"          : True,
        'total_time': []
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 64, 12), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)

    def model_structure(self, model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  # 如果是浮点数就是4

        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)
        
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = Esonarnet(num_classes = self.num_classes)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        # self.model_structure(self.net)
        # print(11)
        self.net = self.net.cuda()
        print('{} model, and classes loaded.'.format(self.model_path))

    def get_miou_png(self, image, png_image, image_id):
        origin_image = image.copy()
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        png_image, nw, nh = resize_image(png_image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        png_image   = np.expand_dims(np.transpose(preprocess_input(np.array(png_image, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            png_image = torch.from_numpy(png_image)
            if self.cuda:
                images = images.cuda()
                png_image = png_image.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            # images = images.repeat(32, 1, 1, 1)
            
            s=time.time()
            pr = self.net(images,images)
            
            self.total_time.append((time.time() - s))
            pr = pr[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

            out1 = pr
            pr = out1.argmax(axis=-1)

        if not os.path.exists('./miou_out/visual/'):
            os.makedirs('./miou_out/visual/')
            
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        
        for c in range(self.num_classes):
            index = np.where(pr[:, :] == c)
            if index[0].size:
                seg_img[index[0], index[1], :] = self.colors[c]
           
        cv2.imwrite('./miou_out/visual/' + image_id + '.png', seg_img)
        image = Image.fromarray(np.uint8(pr))
        return image
