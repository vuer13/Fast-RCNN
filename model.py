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
    
epochs = 10
for epoch in range(epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()
    
    model_path = f'./model/fastrcnn_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")