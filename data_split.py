import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split

# Directories for cards
dataset_root = 'data'
all_images_dir = os.path.join(base_dir, "cards")
coco_json_path = os.path.join(base_dir, "annotations.json")

# Load json annotations
with open(coco_json_path, "r") as f:
    # coco now stores json f, which was opened
    coco = json.load(f)
    
# Names of categories in json
images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Split images into 70%, 15%, 15%
train_imgs, temp_imgs = train_test_split(images, train_size=0.7, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}

