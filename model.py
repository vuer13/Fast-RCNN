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
from torchvision.ops import box_iou

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# import class from another file
from coco import CocoTransform

LABEL_NAMES = {
    1: "name",
    2: "card number",
    3: "card series",
    4: "team name",
    5: "card type"
}

# dataset class
def get_coco_dataset(img_dir, annotation_file):
    return CocoDetection(root=img_dir, annFile=annotation_file, transforms=CocoTransform())

# Getting the model and loading with ResNet50 backbone
def get_model(num_classes):
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pre_trained=True)
    
    # Number of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace with new head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

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
        images = list(image.to(device) for image in images)
        
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
    total_iou, count = 0.0, 0
    
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
                if len(pred_boxes) == 0:
                    continue
                target_boxes = torch.tensor([obj['bbox'] for obj in target], device=device, dtype=torch.float32)
                target_boxes[:, 2:] += target_boxes[:, :2]  # xywh -> xyxy
                
                # Compute IoUs between predicted and target boxes
                ious = box_iou(pred_boxes, target_boxes)
                match_ious = ious.max(dim=1).values
                total_iou += match_ious.mean().item()
                count += 1
    
    model.train()
    mean_iou = total_iou / max(count, 1)
    print(f"Validation mIoU: {mean_iou:.4f}")
    return mean_iou

# Visualization function
def visualize_preds(model, dataset, device, idx=0, label_names=None, threshold=0.5, save_dir='./output'):
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
            
            class_name = label_names.get(label.item(), str(label.item())) if label_names else str(label.item())
            
            # Draw a bounding box around the detected object
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Labels
            cv2.putText(img_np, f"{class_name}:{score:.2f}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    save_path = os.path.join(save_dir, f"pred_{idx}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {save_path}")
    model.train() # Back to train mode
    
def train_model(model, optimizer, scheduler, train_loader, val_loader, device, start_epoch, epochs):
    best_iou = -1
    for epoch in range(start_epoch, epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()
    
        if epoch % 2 == 0:
            mean_iou = evaluate(model, val_loader, device)
            print(f"Validation mIoU (epoch {epoch}): {mean_iou:.4f}")
            
            coco_preds = generate_coco_preds(model, val_loader, device)
            ap = evaluate_coco_ap('./data/val/annotations.json', coco_preds)
            print(f"Validation mAP (epoch {epoch}): {ap:.4f}")
            
            visualize_preds(model, val_loader.dataset, device, idx=random.randint(0, len(val_loader.dataset) - 1), label_names=LABEL_NAMES, save_dir='./output/training', threshold=0.5)
            
        if mean_iou > best_iou:
            print(f"Best Epoch: {epoch}")
            best_iou = mean_iou
            save_checkpoint(model, optimizer, scheduler, epoch, "./model/best_model.pth")
    
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, f'./model/fastrcnn_epoch_{epoch + 1}.pth')
        
        save_checkpoint(model, optimizer, scheduler, epoch, './model/checkpoint_latest.pth')
    
def test_model(model, dataset, device, label_names=None, save_dir='./output/test_preds', threshold=0.5):
    model.eval() # Evaluation mode
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(len(dataset)):
        visualize_preds(model, dataset, device, idx=idx, label_names=label_names, threshold=threshold, save_dir=save_dir)
    model.train() # Back to train mode
    
def generate_coco_preds(model, data_loader, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            # images to device, run inference
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Ground truth image_id
                image_id = target[0]['image_id']
                
                # Predictions
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    
                    # append COCO prediction
                    result.append({
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [x1, y1, width, height],
                        "score": float(score)
                    })
    
    return results

def evaluate_coco_ap(annotations, predictions):
    coco_gt = COCO(annotations) # truth annoations
    coco_dt = coco_gt.loadRes(predictions) # model predictions
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    
    # Evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0]  # mAP (AP@0.5:0.95)
            
if __name__ == "__main__":
    num_classes = 6  # 5 classes + background
    start_epoch = 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Loading datasets
    train_dataset = get_coco_dataset(img_dir='./data/train/images', annotation_file='./data/train/annotations.json')
    val_dataset = get_coco_dataset(img_dir='./data/val/images', annotation_file='./data/val/annotations.json')

    train_loader=DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader=DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # intializing model
    model = get_model(num_classes)

    # model to gpu if possible (may train in notebook)
    model.to(device)

    # Optimizer and LR
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0005) # to update weights in backward pass
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    checkpoint_path = './model/checkpoint_latest.pth'  

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        start_epoch = loadcheckpoint(model, optimizer, lr_scheduler, checkpoint_path, device)
    else:
        print("Training New Model")
        
    epochs = start_epoch + 10  # Train for additional 10 epochs from start_epoch
        
    train_model(model, optimizer, lr_scheduler, train_loader, val_loader, device, start_epoch, epochs)
    
    final_iou = evaluate(model, val_loader, device)
    print(f"Validation mIoU: {final_iou:.4f}")
    
    test_dataset = get_coco_dataset(img_dir='./data/test/images', annotation_file='./data/test/annotations.json')
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    final_test_iou = evaluate(model, test_loader, device)
    print(f"Test mIoU: {final_test_iou:.4f}")
    
    coco_test_preds = generate_coco_preds(model, test_loader, device)
    final_test_ap = evaluate_coco_ap('./data/test/annotations.json', coco_test_preds)
    print(f"Test mAP: {final_test_ap:.4f}")
    
    test_model(model, test_dataset, device, label_names=LABEL_NAMES)