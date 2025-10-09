import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split

# Directories for cards
dataset_root = 'data'
all_images_dir = os.path.join(dataset_root, "cards")
coco_json_path = os.path.join(dataset_root, "annotations.json")

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

# Function to filter the annotations
def filter_annotations(image_subset):
    image_ids = {img["id"] for img in image_subset}
    return [ann for ann in annotations if ann["image_id"] in image_ids]

# Splitting images and annotations
for split_name, subset_imgs in splits.items():
    # Where all info goes per category
    split_dir = os.path.join(dataset_root, split_name)
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    subset_data = {
        "images": subset_imgs,
        "annotations": filter_annotations(subset_imgs),
        "categories": categories
    }

    # Write new annotations.json
    with open(os.path.join(split_dir, "annotations.json"), "w") as f:
        json.dump(subset_data, f, indent=2)

    # Copy images & duplicate
    for img in subset_imgs:
        src = os.path.join(all_images_dir, img["file_name"])
        dst = os.path.join(img_dir, img["file_name"])
        shutil.copy2(src, dst) 