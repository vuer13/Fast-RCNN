import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split

# Directories for cards
dataset_root = 'data'
all_images_dir = os.path.join(base_dir, "cards")
coco_json_path = os.path.join(base_dir, "annotations.json")