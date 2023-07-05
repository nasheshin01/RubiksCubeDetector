from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from tqdm.auto import tqdm
 
import os
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import time
import torchvision.transforms as transforms
import torch
from PIL import Image

if __name__ == '__main__':
    ROOT_DIR = 'YoloDataset'
    train_imgs_dir = 'train/images'
    train_labels_dir = 'train/labels'
    val_imgs_dir = 'valid/images'
    val_labels_dir = 'valid/labels'
    test_imgs_dir = 'test/images'
    test_labels_dir = 'test/labels'
    classes = ['cube']

    dataset_params = {
        'data_dir':ROOT_DIR,
        'train_images_dir':train_imgs_dir,
        'train_labels_dir':train_labels_dir,
        'val_images_dir':val_imgs_dir,
        'val_labels_dir':val_labels_dir,
        'test_images_dir':test_imgs_dir,
        'test_labels_dir':test_labels_dir,
        'classes':classes 
    }

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['test_images_dir'],
            'labels_dir': dataset_params['test_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':1
        }
    )

    # best_model = models.get('yolo_nas_s',
    #                         num_classes=len(dataset_params['classes']),
    #                         checkpoint_path="YoloTrainPart\\checkpoints\\yolo_nas_s\\ckpt_best.pth")
    best_model = models.get('yolo_nas_s',
                            num_classes=len(dataset_params['classes']),
                            checkpoint_path="ResultModels\\cube_detector_v1.pth")
    
    # models.convert_to_onnx(model=best_model, input_shape=(3,640,640), out_path="yolo_nas_s.onnx")


    trainer = Trainer(
            experiment_name="yolo_nas_s", 
            ckpt_root_dir="YoloTrainPart\\checkpoints")
    test_out = trainer.test(model=best_model,
                 test_loader=test_data,
                 test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                       top_k_predictions=300, 
                                                       num_cls=len(dataset_params['classes']), 
                                                       normalize_targets=True, 
                                                       post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                              nms_top_k=1000, 
                                                                                                              max_predictions=300,                                                                              
                                                                                                              nms_threshold=0.7)))
    
    print(test_out)