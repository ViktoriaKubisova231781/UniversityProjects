# app_frontend_local/scripts/preprocess_feedback.py

import shutil
import subprocess
from pathlib import Path

# Paths
feedback_dir = Path("./app_frontend_local/feedback_data")
raw_images_dir = Path("./app_frontend_local/datasets/images")
raw_masks_dir = Path("./app_frontend_local/datasets/masks")
patch_image_dir = Path("./app_frontend_local/datasets/patches/images")
patch_mask_dir = Path("./app_frontend_local/datasets/patches/masks")

train_image_output = Path("./app_frontend_local/original_dataset/images/train")
train_mask_output = Path("./app_frontend_local/original_dataset/masks/train")

# Step 1: Create temp preprocessing folders
raw_images_dir.mkdir(parents=True, exist_ok=True)
raw_masks_dir.mkdir(parents=True, exist_ok=True)
patch_image_dir.mkdir(parents=True, exist_ok=True)
patch_mask_dir.mkdir(parents=True, exist_ok=True)

# Step 2: Copy and rename feedback files to preprocessing inputs
for file in feedback_dir.glob("*.png"):
    shutil.copy(file, raw_images_dir / file.name)

for file in feedback_dir.glob("*.tif"):
    base = file.stem.replace("_mask", "")  # rename: image_mask.tif → image_root_mask.tif
    new_name = f"{base}_root_mask.tif"
    shutil.copy(file, raw_masks_dir / new_name)

# Step 3: (Optional) Clear feedback_data
shutil.rmtree(feedback_dir)

# Step 4: Run patching pipeline
subprocess.run([
    "python", "app_frontend_local/scripts/main.py",
    "--images_dir", str(raw_images_dir),
    "--masks_dir", str(raw_masks_dir),
    "--output_images_dir", str(patch_image_dir),
    "--output_masks_dir", str(patch_mask_dir),
    "--patch_size", "256",
    "--step", "128"
], check=True)

# Step 5: Move patched data to train set
train_image_output.mkdir(parents=True, exist_ok=True)
train_mask_output.mkdir(parents=True, exist_ok=True)

for patch in patch_image_dir.glob("*.png"):
    shutil.move(str(patch), train_image_output / patch.name)

for patch in patch_mask_dir.glob("*.tif"):
    shutil.move(str(patch), train_mask_output / patch.name)

# (Optional) Clean patch folders
shutil.rmtree(patch_image_dir.parent)
