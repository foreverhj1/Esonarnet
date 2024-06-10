import os

from PIL import Image
from tqdm import tqdm

from utils.config import Model
from utils.utils_metrics import compute_mIoU, show_results

import time
import numpy as np

if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    num_classes     = 12
    name_classes    = ["background", "bottle", "can",'chain','drink-carton','hook','propeller','shampoo-bottle','standing-bottle','tire','valve','wall']
    VOCdevkit_path  = 'VOCdevkit'
    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        model = Model()
        print("Load model done.")
        print("Get predict result.")
        #total_time=0
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")
            hog_path = os.path.join(VOCdevkit_path, "VOC2007/HOG/" + image_id + ".png")
            image       = Image.open(image_path)
            png_image  = Image.open(hog_path)
            image       = model.get_miou_png(image, png_image, image_id)
            image.save(os.path.join(pred_dir, image_id + ".png"))

        total_time=model.total_time
        total_time.pop(0)
        print("Get predict result done.")
    
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print(PA_Recall)
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)