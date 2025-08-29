import os, uuid
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLO
import glob, shutil 

register_heif_opener()

input_dir = './dataset'
output_dir = './prepped_data/cards'
os.makedirs(output_dir, exist_ok=True)

MAX_LONG_SIDE = 1600 
count = 1

def save_jpg(src_path, dst_dir, i):
    img = Image.open(src_path)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    
    scale = min(1.0, MAX_LONG_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        
    img = img.convert("RGB")
    
    out_name = f"card_{i:04d}.jpg"
    out_path = os.path.join(dst_dir, out_name)
    img.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
    return out_name

for fname in sorted(os.listdir(input_dir)):
    if fname.lower().endswith((".heic", ".heif", ".png", ".jpg", ".jpeg")):
        src = os.path.join(input_dir, fname)
        new_name = save_jpg(src, output_dir, count)
        count += 1

input_dir= './prepped_data/cards'
output_dir = "./cropped_cards"
        
model = YOLO("yolo.pt")
results = model.predict(source=input_dir, save_crop=True, project=output_dir, name="run1")

crop_dir = os.path.join(output_dir, "run1", "crops", "hockey_card")
final_dir = "./data/cards"
os.makedirs(final_dir, exist_ok=True)

for f in glob.glob(os.path.join(crop_dir, "*.jpg")):
    shutil.move(f, os.path.join(final_dir, os.path.basename(f)))