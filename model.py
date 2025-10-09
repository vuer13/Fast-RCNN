import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

import matplotlib.pyplot as plt
from PIL import Image

# import class from another file
from coco import CocoTransform

# dataset class
def get_coco_dataset(img_dir, annotation_file):
    return CocoDetection(root=img_dir, annFile=annotation_file, transforms=CocoTransform())

# Loading datasets
train_dataset = get_coco_dataset(
    img_dir='./data/train/image',
    annotation_file='./data/train/annotations.json'
)

val_dataset = get_coco_dataset(
    img_dir='./data/val/image',
    annotation_file='./data/val/annotations.json'
)

train_loader=DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader=DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Getting the model and loading with ResNet50 backbone
def get_model(num_classes):
    # TODO