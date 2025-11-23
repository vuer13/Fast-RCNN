# FasterRCNN - Hockey Card Annotator

A fine-tuned Faster R-CNN (ResNet-50 FPN) model trained on a custom hockey card dataset.
The model detects and localizes key elements of a card (e.g., player name, card number, team name, series, and card type) for use in downstream computer vision applications.

Performance:
- mIoU: 0.8607
- mAP (0.5–0.95): 0.7154

### Further Work:
- Integrating semi-supervised learning to leverage unlabeled card images
- Expanding dataset size to improve generalization and class balance