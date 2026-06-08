#Final Integrated CV + PID Pipeline with Fixed Crop Offset and Axis Correction
import tensorflow as tf
import keras.backend as K
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from patchify import patchify, unpatchify
from skimage.morphology import skeletonize
from pid_controller_3D import PIDController3D
from sim_class import Simulation

# --- F1 Metric ---
def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return TP / (Positives + K.epsilon())
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return TP / (Pred_Positives + K.epsilon())
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# --- Load Model ---
print("Loading model...")
model = load_model(r"C:\Users\vikku\OneDrive\Documenten\Buas\2024-25b-fai2-adsai-ViktoriaKubisova231781\datalab_tasks\task5\12_viktoria_231781_unet_model_256px.h5", custom_objects={"f1": f1})

# --- Image Utilities ---
def padder(image, patch_size):
    h, w = image.shape[:2]
    print(f"Original image shape: {h}x{w}")
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w
    top = height_padding // 2
    bottom = height_padding - top
    left = width_padding // 2
    right = width_padding - left
    print(f"Padding: top={top}, bottom={bottom}, left={left}, right={right}")
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right)

def unpadder(image, padding):
    top, bottom, left, right = padding
    print(f"Unpadding with: top={top}, bottom={bottom}, left={left}, right={right}")
    return image[top:image.shape[0] - bottom, left:image.shape[1] - right]

def cropper(image):
    orig = image.shape
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return image, {"original_shape": orig, "used_crop": False}
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, cx - size // 2)
    y0 = max(0, cy - size // 2)
    print(f"Cropping to: x0={x0}, y0={y0}, size={size}")
    cropped = image[y0:y0 + size, x0:x0 + size]
    return cropped, {"original_shape": orig, "used_crop": True, "x_start": x0, "y_start": y0, "crop_size": size}

def uncropper(cropped_img, info):
    canvas = np.zeros(info["original_shape"][:2], dtype=cropped_img.dtype)
    if not info.get("used_crop", False):
        return cropped_img
    x0, y0, sz = info["x_start"], info["y_start"], info["crop_size"]
    print(f"Uncropping to canvas at x0={x0}, y0={y0}, size={sz}")
    canvas[y0:y0 + sz, x0:x0 + sz] = cropped_img
    return canvas

# --- Main CV Pipeline ---
def run_full_pipeline(image_path, model, return_crop_info=False):
    print(f"\nRunning full pipeline on: {image_path}")
    image = cv2.imread(image_path, 0)
    name = os.path.basename(image_path)
    petri, crop_info = cropper(image)
    padded, pad_info = padder(petri, 256)
    print(f"Petri cropped shape: {petri.shape}, padded shape: {padded.shape}")
    patches = patchify(padded, (256, 256), step=128)
    reshaped = patches.reshape(-1, 256, 256, 1) / 255.0
    print(f"Number of patches: {reshaped.shape[0]}")
    preds = model.predict(reshaped)
    preds = preds.squeeze().reshape(patches.shape[0], patches.shape[1], 256, 256)
    mask = unpatchify(preds, padded.shape)
    mask = unpadder(mask, pad_info)
    mask = uncropper(mask, crop_info)

    binary = (mask > 0.1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=5)
    binary = cv2.erode(binary, kernel, iterations=3)

    x0 = crop_info['x_start']
    x1 = min(x0 + crop_info['crop_size'], binary.shape[1])
    top_crop_offset = int(binary.shape[0] * 0.15)
    crop = binary[top_crop_offset:, x0:x1]
    print(f"Final cropped segment shape: {crop.shape}")

    h, w = crop.shape
    bw = w // 5
    masks = [np.zeros_like(crop) for _ in range(5)]
    _, lbls, stats, ctrs = cv2.connectedComponentsWithStats(crop)

    for i in range(1, len(stats)):
        x, y, cw, ch, area = stats[i]
        cx, cy = ctrs[i]
        if area < 100: continue
        if ch / cw < 1.5 or y > int(h * 0.3): continue
        if area < 1200:
            angle = np.degrees(np.arctan2(ch, cw))
            if angle < 65 or x < 250 or (x+cw) > (w-250): continue
        idx = min(int(cx // bw), 4)
        cmask = (lbls == i).astype(np.uint8)
        if np.count_nonzero(masks[idx]) < area:
            masks[idx] = cmask

    tips = []
    for i, mask in enumerate(masks):
        if np.count_nonzero(mask) == 0:
            tips.append(None)
            continue
        _, lbls, stats, _ = cv2.connectedComponentsWithStats(mask)
        if len(stats) <= 1:
            tips.append(None)
            continue
        largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        x, y, w, h, _ = stats[largest]
        comp = (lbls == largest).astype(np.uint8)[y:y+h, x:x+w]
        skel = skeletonize(comp)
        coords = np.argwhere(skel)
        if len(coords) == 0:
            tips.append(None)
            continue
        bottom = coords[np.argmax(coords[:, 0])]
        tip = (bottom[1] + x + x0, bottom[0] + y + top_crop_offset)
        print(f"Detected tip {i}: {tip}")
        tips.append(tip)

    if return_crop_info:
        return tips, crop_info
    return tips

# --- Coordinate Conversion ---
def pixel_to_robot_coords(coords, img_shape, robot_origin, plate_size_mm=150):
    height_px = img_shape[0]
    scale = plate_size_mm / height_px
    print(f"Pixel-to-robot scale: {scale} mm/px based on height_px={height_px}")
    result = []
    for idx, c in enumerate(coords):
        if c is None:
            result.append(None)
            continue
        y_px, x_px = c
        x_m = x_px * scale / 1000.0
        y_m = y_px * scale / 1000.0
        corrected_x = robot_origin[0] + y_m 
        corrected_y = robot_origin[1] + x_m
        print(f"Tip {idx}: Pixel ({x_px}, {y_px}) -> Robot ({corrected_x}, {corrected_y})")
        result.append([corrected_x, corrected_y, 0.17])
    return result

# --- Reverse Mapping ---
def robot_to_pixel_coords(robot_coords, img_shape, robot_origin, plate_size_mm=150):
    height_px, width_px = img_shape[:2]
    scale = plate_size_mm / height_px
    print(f"Robot-to-pixel scale: {scale} mm/px")
    result = []
    for idx, coord in enumerate(robot_coords):
        if coord is None:
            result.append(None)
            continue
        dx = coord[0] - robot_origin[0]
        dy = coord[1] - robot_origin[1]
        x_px = int((dx * 1000.0) / scale)
        y_px = int((dy * 1000.0) / scale)
        print(f"Drop {idx}: Robot ({coord[0]}, {coord[1]}) -> Pixel ({x_px}, {y_px})")
        result.append((x_px, y_px))
    return result

import csv  

# --- Simulation and Controller Execution ---
sim = Simulation(num_agents=1, render=True)
try:
    print("[INFO] Getting initial simulation states...")
    states = sim.get_states()
    robot_id = int(list(states.keys())[0].split("_")[-1])
    print(f"[INFO] Detected robot ID: {robot_id}")

    image_path = sim.get_plate_image()
    print(f"[INFO] Plate image path received: {image_path}")
    
    root_tips, crop_info = run_full_pipeline(image_path, model, return_crop_info=True)
    img = cv2.imread(image_path, 0)

    indexed_tips = [(i, tip) for i, tip in enumerate(root_tips) if tip is not None]
    sorted_tips = sorted(indexed_tips, key=lambda x: x[1][0])
    sorted_root_tips = [None] * len(root_tips)
    for i, (_, tip) in zip(range(len(indexed_tips)), sorted_tips):
        sorted_root_tips[i] = tip

    print(f"[INFO] Sorted root tips: {sorted_root_tips}")

    plate_origin = [0.10775, 0.062, 0.057]
    crop_shape = (crop_info["crop_size"], crop_info["crop_size"])

    # === Adjust tips from full-image space to cropped image space
    adjusted_tips = []
    for tip in sorted_root_tips:
        if tip is None:
            adjusted_tips.append(None)
        else:
            x_full, y_full = tip
            x_crop = x_full - crop_info["x_start"]
            y_crop = y_full - crop_info["y_start"]  # FIXED: Don't subtract top_crop_offset again
            adjusted_tips.append((y_crop, x_crop))  # Use (y, x) for compatibility

    print(f"[DEBUG] Adjusted tips for cropped image space: {adjusted_tips}")

    # === Now use correct cropped image shape with adjusted tip positions
    coords_3d = pixel_to_robot_coords(adjusted_tips, crop_shape, plate_origin)
    print(f"[INFO] Converted robot coordinates: {coords_3d}")

    controller = PIDController3D([18, 18, 18], [0, 0, 0], [0, 0, 0])
    
    dt = 1.0
    reached_positions = []
    csv_rows = [
        ["Root Index", "Target_X", "Target_Y", "Target_Z",
         "Actual_X", "Actual_Y", "Actual_Z",
         "Delta_X", "Delta_Y", "Delta_Z",
         "Euclidean_Distance", "Drop_Pixel_X", "Drop_Pixel_Y"]
    ]

    trajectories = [[] for _ in coords_3d]
    for idx, goal in enumerate(coords_3d):
        print(f"\n[INFO] === Starting movement for root {idx} ===")
        if goal is None:
            print(f"[WARN] Skipping root {idx} (no goal coordinates).")
            reached_positions.append(None)
            continue

        controller.reset()
        curr = np.array(sim.get_pipette_position(robot_id))
        trajectories[idx].append(curr.copy()) 
        print(f"[INFO] Initial pipette position: {curr}")

        # === Main Precision Phase ===
        stable_count = 0
        required_stable_frames = 10
        threshold = 0.0008

        for step in range(200):
            err = np.array(goal) - curr
            distance = np.linalg.norm(err)
            max_speed = 0.2 if distance < 0.01 else 1.0

            out = controller.compute(err, dt)
            out = np.clip(out, -max_speed, max_speed)

            sim.run([np.concatenate([out, [0]])])
            curr = np.array(sim.get_pipette_position(robot_id))
            trajectories[idx].append(curr.copy())

            if step % 10 == 0 or distance < threshold:
                print(f"[DEBUG] Step {step}: distance = {distance:.6f}, position = {curr}")

            if distance < threshold:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= required_stable_frames:
                print("[INFO] Tip reached with sufficient stability. Executing drop...")
                for _ in range(15):
                    sim.run([[0, 0, 0, 0]])
                sim.run([[0, 0, 0, 1]])
                for _ in range(30):
                    sim.run([[0, 0, 0, 0]])
                break

        reached_positions.append(curr.tolist())
        print(f"[INFO] Final reached position for root {idx}: {curr.tolist()}")

        delta = np.array(curr) - np.array(goal)
        distance = np.linalg.norm(delta)

        drop_px = robot_to_pixel_coords([curr.tolist()], crop_shape, plate_origin)[0]
        drop_px_x, drop_px_y = drop_px if drop_px else (None, None)

        csv_rows.append([
            idx,
            round(goal[0], 5), round(goal[1], 5), round(goal[2], 5),
            round(curr[0], 5), round(curr[1], 5), round(curr[2], 5),
            round(delta[0], 5), round(delta[1], 5), round(delta[2], 5),
            round(distance, 5),
            drop_px_x, drop_px_y
        ])

    # === Save CSV log ===
    with open("droplet_log_13.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_rows)
    print("[INFO] CSV log saved as 'droplet_log_13.csv'")

    # === Overlay Debug Image ===
    print("[INFO] Generating overlay debug image...")
    debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for i, tip in enumerate(sorted_root_tips):
        if tip:
            x, y = int(tip[0]), int(tip[1])
            cv2.circle(debug_img, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(debug_img, f"R{i+1}", (x-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    reached_pixel_coords = robot_to_pixel_coords(reached_positions, img.shape, plate_origin)
    for i, pt in enumerate(reached_pixel_coords):
        if pt:
            x, y = pt
            cv2.circle(debug_img, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(debug_img, f"D{i+1}", (x-20, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite("root_tip_overlay_debug_13.png", debug_img)
    print("[INFO] Saved overlay image: 'root_tip_overlay_debug_13.png'")

finally:
    sim.close()
    print("[INFO] Simulation closed safely.")

import pandas as pd
import matplotlib.pyplot as plt

# === Load performance log ===
csv_path = "droplet_log_13.csv"
df = pd.read_csv(csv_path)

# === Plot 1: Target vs Actual X Position ===
plt.figure(figsize=(8, 5))
plt.plot(df["Root Index"], df["Target_X"], 'o-', label="Target X")
plt.plot(df["Root Index"], df["Actual_X"], 'x--', label="Actual X")
plt.title("Target vs Actual X Position")
plt.xlabel("Root Index")
plt.ylabel("X Position (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_x_position.png")
plt.show()

# === Plot 2: Target vs Actual Y Position ===
plt.figure(figsize=(8, 5))
plt.plot(df["Root Index"], df["Target_Y"], 'o-', label="Target Y")
plt.plot(df["Root Index"], df["Actual_Y"], 'x--', label="Actual Y")
plt.title("Target vs Actual Y Position")
plt.xlabel("Root Index")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_y_position.png")
plt.show()

# === Plot 3: Final Euclidean Distance from Target ===
plt.figure(figsize=(8, 5))
plt.plot(df["Root Index"], df["Euclidean_Distance"], 's-', color="purple")
plt.title("Final Distance to Target per Root")
plt.xlabel("Root Index")
plt.ylabel("Distance (m)")
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_distance.png")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# === Per-root step-by-step XYZ trajectory plots ===
for idx, (target, path) in enumerate(zip(coords_3d, trajectories)):
    if target is None or not path:
        continue

    path = np.array(path)
    steps = list(range(len(path)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- X-axis ---
    axes[0].plot(steps, path[:, 0], color="red", label="Observation X")
    axes[0].axhline(target[0], color="green", linestyle="--", label="Target X")
    axes[0].set_title("X-axis")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Position")
    axes[0].legend()
    axes[0].grid(True)

    # --- Y-axis ---
    axes[1].plot(steps, path[:, 1], color="red", label="Observation Y")
    axes[1].axhline(target[1], color="green", linestyle="--", label="Target Y")
    axes[1].set_title("Y-axis")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Position")
    axes[1].legend()
    axes[1].grid(True)

    # --- Z-axis ---
    axes[2].plot(steps, path[:, 2], color="red", label="Observation Z")
    axes[2].axhline(target[2], color="green", linestyle="--", label="Target Z")
    axes[2].set_title("Z-axis")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Position")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"Root {idx} - PID Trajectory Tracking", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"trajectory_xyz_root_{idx}.png")
    plt.show()


# === Performance Statistics ===
step_counts = [len(path) for path in trajectories if path]
avg_steps = np.mean(step_counts) if step_counts else 0
print(f"\n[STATS] Average steps to reach target: {avg_steps:.2f}")

# === Accuracy (based on final Euclidean distance) ===
threshold = 0.001  # 1mm tolerance
accurate_hits = df["Euclidean_Distance"] < threshold
accuracy = accurate_hits.sum() / len(accurate_hits) * 100
print(f"[STATS] Accuracy (within {threshold*1000:.1f}mm): {accuracy:.1f}%")

# Optional: print individual distances
print("\n[STATS] Final distances per root:")
for idx, dist in enumerate(df["Euclidean_Distance"]):
    print(f"  Root {idx}: {dist:.6f} m")
