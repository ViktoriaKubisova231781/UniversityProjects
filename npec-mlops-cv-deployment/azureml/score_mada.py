import os
import io
import json
import base64
import time
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
import cv2
from patchify import patchify, unpatchify

# ─── Logging Setup ────────────────────────────────────────
LOG_PATH = os.path.join("/tmp", "mada_inference_log.txt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("azureml")

if not logger.hasHandlers():
    file_handler = logging.FileHandler(LOG_PATH)
    logger.addHandler(file_handler)

# ─── Globals ──────────────────────────────────────────────
PATCH_SIZE = 256
model = None

# ─── Custom Metrics ───────────────────────────────────────
def f1(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    pre = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (pre + K.epsilon())
    recall = tp / (pos + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        pt_1 = y_true * K.log(y_pred)
        pt_0 = (1 - y_true) * K.log(1 - y_pred)
        return -alpha * K.pow(1 - y_pred, gamma) * pt_1 - (1 - alpha) * K.pow(y_pred, gamma) * pt_0
    return focal_loss_fixed

# ─── Model Init ───────────────────────────────────────────
def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "madalina_221989_cnn_custom_model_deploy_256px.h5")
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={"f1": f1, "focal_loss_fixed": focal_loss()})
    logger.info("Model loaded successfully.")

# ─── Helper Functions ─────────────────────────────────────
def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("L"))

def padder(image, patch_size):
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w
    top = height_padding // 2
    bottom = height_padding - top
    left = width_padding // 2
    right = width_padding - left
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right)

def unpadder(padded, pads):
    top, bottom, left, right = pads
    h, w = padded.shape[:2]
    return padded[top : h - bottom, left : w - right]

def cropper(image):
    orig_shape = image.shape
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image, {"original_shape": orig_shape, "used_crop": False}
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    xs = max(cx - size // 2, 0)
    ys = max(cy - size // 2, 0)
    cropped = image[ys : ys + size, xs : xs + size]
    return cropped, {
        "original_shape": orig_shape,
        "used_crop": True,
        "x_start": xs,
        "y_start": ys,
        "crop_size": size,
    }

def uncropper(cropped, info):
    if not info.get("used_crop", False):
        return cropped
    h0, w0 = info["original_shape"]
    canvas = np.zeros((h0, w0), dtype=cropped.dtype)
    xs, ys, sz = info["x_start"], info["y_start"], info["crop_size"]
    canvas[ys : ys + sz, xs : xs + sz] = cropped
    return canvas

def threshold_mask(raw_mask, threshold=0.3):
    return (raw_mask > threshold).astype(np.uint8)

def morphological_closing(binary, kernel_size=(3, 3), dilate_iter=5, erode_iter=3, kernel_shape=cv2.MORPH_ELLIPSE):
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    dilated = cv2.dilate(binary, kernel, iterations=dilate_iter)
    closed = cv2.erode(dilated, kernel, iterations=erode_iter)
    return closed.astype(np.float32)

# ─── Inference ────────────────────────────────────────────
def run(raw_data):
    try:
        logger.info("[RUN] Inference started.")
        start_time = time.time()

        input_data = json.loads(raw_data)
        base64_image = input_data["data"]
        image_bytes = base64.b64decode(base64_image)

        img = load_image_from_bytes(image_bytes)
        logger.info(f"Image shape: {img.shape}")

        cropped, crop_info = cropper(img)
        padded, pad_info = padder(cropped, PATCH_SIZE)

        patches = patchify(padded, (PATCH_SIZE, PATCH_SIZE), step=128)
        n_h, n_w, _, _ = patches.shape
        batch = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 1).astype("float32") / 255.0

        preds = model.predict(batch, verbose=0).squeeze()
        preds = preds.reshape(n_h, n_w, PATCH_SIZE, PATCH_SIZE)

        full_padded = unpatchify(preds, padded.shape)
        unpadded = unpadder(full_padded, pad_info)
        mask = uncropper(unpadded, crop_info)

        binary_mask = threshold_mask(mask)
        closed_mask = morphological_closing(binary_mask)

        mask_img = Image.fromarray((closed_mask * 255).astype(np.uint8))
        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")
        encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")

        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} sec")

        return json.dumps({
            "predicted_mask_base64": encoded_mask,
            "crop_info": crop_info,
            "pad_info": pad_info,
            "inference_time_sec": inference_time
        })

    except Exception as e:
        logger.error(f"[ERROR] Inference failed: {str(e)}")
        return json.dumps({"error": str(e)})
