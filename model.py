import os
import cv2
import numpy as np
import random

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
    img_dir='./data/train/images',
    annotation_file='./data/train/annotations.json'
)

val_dataset = get_coco_dataset(
    img_dir='./data/val/images',
    annotation_file='./data/val/annotations.json'
)

train_loader=DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader=DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Getting the model and loading with ResNet50 backbone
def get_model(num_classes):
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pre_trained=True)
    
    # Number of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace with new head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# intializing model
num_classes = 6  # 5 classes + background
model = get_model(num_classes)

# model to gpu if possible (may train in notebook)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and LR
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0005) # to update weights in backward pass
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Saving a checkpoint to save model state
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)
    print("Checkpoint saved at", path)

# To load a checkpoint
def loadcheckpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print("Loaded checkpoint from", path, "at epoch", start_epoch)
    return start_epoch

# Training function - trains one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    losses_per_epoch = []
    
    for images, targets in data_loader:
        # Move images to device
        images = list(image.to(device) for image in images)\
        
        # Process targets - classnames and bounding box 
        processed_targets = []
        valid_images = []
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            for obj in target:
                # get bounding box 
                bbox = obj['bbox']
                x, y, w, h = bbox

                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h]) # x_min, y_min, x_max, y_max
                    labels.append(obj['category_id'])
            
            # Only process is there are boxes in the image
            if boxes:
                processed_target = {
                    'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
                    'labels': torch.tensor(labels, dtype=torch.int64).to(device)
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i]) # valid images only
                
        # skip if no valid targets
        if not processed_targets:
            continue
        
        images = valid_images
        
        # Forward pass - training the model
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        losses_per_epoch.append(losses.item())
        
    print(f"Epoch {epoch}, Loss: {losses.item()}, Avg Loss: {np.mean(losses_per_epoch):.4f}")
    
# Evaluation
def evaluate(model, data_loader, device, threshold=0.5):
    model.eval() # Evaluation mode
    total_iou, count = 0, 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            
            # Loop through image and predictions
            for pred, target in zip(preds, targets):
                if len(pred['boxes']) == 0 or len(target) == 0:
                    continue
                
                # Keep predictions if > threshold
                pred_boxes = pred['boxes'][pred['scores'] > threshold]
                target_boxes = torch.tensor([obj['bbox'] for obj in target], device=device)
                target_boxes[:, 2:] += target_boxes[:, :2]  # xywh -> xyxy
                
                # Compute IoUs between predicted and target boxes
                ious = box_iou(pred_boxes, target_boxes)
                total_iou += ious.max(dim=1)[0].sum().item() if ious.numel() > 0 else 0
                count += 1
    
    model.train()
    mean_iou = total_iou / max(count, 1)
    print(f"Validation mIoU: {mean_iou:.4f}")
    return mean_iou

# Visualization function
def visualize_preds(model, dataset, device, idx=0, threshold=0.5, save_dir='./output'):
    model.eval() # Evaluation mode
    os.makedirs(save_dir, exist_ok=True)
    
    # Load image and target from dataset
    image, target = dataset[idx]
    
    with torch.no_grad():
        # Model inference on single image
        pred = model([image.to(device)])[0]
        
    # Convert to image tensor to numpy for visualization
    img_np = np.array(F.to_pil_image(image))
    
    # Loop through predictions and draw boxes
    for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            
            # Draw a bounding box around the detected object
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Labels
            cv2.putText(img_np, f"{label.item()}:{score:.2f}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    save_path = os.path.join(save_dir, f"pred_{idx}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {save_path}")
    model.train() # Back to train mode

epochs = 10
start_epoch = 0

if os.path.exists('./model/checkpoint_latest.pth'):
    start_epoch = loadcheckpoint(model, optimizer, lr_scheduler, './model/checkpoint_latest.pth', device)

for epoch in range(start_epoch, epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()
    
    if epoch % 2 == 0:
        mean_iou = evaluate(model, val_loader, device)
        print(f"Validation mIoU (epoch {epoch}): {mean_iou:.4f}")
        visualize_preds(model, val_dataset, device, idx=random.randint(0, len(val_dataset) - 1))
    
    save_checkpoint(model, optimizer, lr_scheduler, epoch, f'./model/fastrcnn_epoch_{epoch + 1}.pth')
    save_checkpoint(model, optimizer, lr_scheduler, epoch, './model/checkpoint_latest.pth')