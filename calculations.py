import torch
import numpy as np
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def compute_confusion_matrix(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5, num_classes=6):
    model.eval()
    
    all_gt = [] # Ground truth values
    all_pred = [] # Predicted values
    
    with torch.no_grad():
        # Looping over dataset
        print("Starting confusion matrix computation")
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            preds = model(images)
            
            # Process each image
            for pred, t in zip(preds, targets):
                
                # Filter predictions
                keep = pred['scores'] > score_threshold # only keep boxes with score > score_threshold
                pred_boxes = pred['boxes'][keep] # Filtered boxes
                pred_labels = pred['labels'][keep].cpu().numpy() # Filtered labels to numpy arrays
                
                # Ground truth boxes and labels
                gt_boxes = []
                gt_labels = []
                
                for obj in t:
                    x, y, w, h = obj['bbox']
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_labels.append(obj['category_id'])
                    
                # Convert to tensors    
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
                gt_labels = np.array(gt_labels, dtype=int)

                # All FN if no predictions
                if len(pred_boxes) == 0:
                    for g in gt_labels:
                        all_gt.append(g)
                        all_pred.append(0)
                    continue
                
                # IoU matrix
                ious = box_iou(pred_boxes, gt_boxes)
                max_ious, gt_id = ious.max(dim=1)
                used_gt = set()
                
                # Processing each prediction
                for i, iou in enumerate(max_ious):
                    p_label = int(pred_labels[i])
                    
                    if iou >= iou_threshold:
                        # Check if ground truth is unused: TP
                        g_i = gt_id[i].item()
                        g_label = gt_labels[g_i]

                        if g_i not in used_gt:
                            # True positive
                            all_gt.append(g_label)
                            all_pred.append(p_label)
                            used_gt.add(g_i)
                        else:
                            # Duplicate detection: FP
                            all_gt.append(0)
                            all_pred.append(p_label)
                    else:
                        # IoU too low: FP
                        all_gt.append(0)
                        all_pred.append(p_label)
                        
                # FN: GT not matched by prediction
                for g_i, g_label in enumerate(gt_labels):
                    if g_i not in used_gt:
                        all_gt.append(g_label) 
                        all_pred.append(0)   
              
    print("Finished computing confusion matrix")          
    cm = confusion_matrix(all_gt, all_pred, labels=list(range(num_classes)))
    return cm

def plot_confusion_matrix_heatmap(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))

    # Heatmap visualization
    sns.heatmap(cm,
                annot=True,     
                fmt='d',       
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Faster R-CNN Confusion Matrix")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")
        
    plt.show()
    
def main(model, test_loader, device):
    class_names = [
        "background",
        "name",
        "card number",
        "card series",
        "team name",
        "card type"
    ]

    cm = compute_confusion_matrix(
        model,
        test_loader,
        device,
        iou_threshold=0.5,
        score_threshold=0.5,
        num_classes=6
    )
    
    # Save as numpy file
    np.save("plots/confusion_matrix.npy", cm)
    print("Saved matrix to plots/confusion_matrix.npy")

    # Save as txt file
    np.savetxt("plots/confusion_matrix.txt", cm, fmt="%d")
    print("Saved matrix as txt to plots/confusion_matrix.txt")

    plot_confusion_matrix_heatmap(cm, class_names, save_path='./plots/confusion_matrix.png')
