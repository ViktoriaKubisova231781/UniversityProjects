# ─── Imports ─────────────────────────────────────────────────────────

import tensorflow as tf
from tensorflow.keras import backend as K
from azureml.core import Workspace, Model
import os
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


# ─── Model functions ─────────────────────────────────────────────────

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_model(model_path: str) -> tf.keras.Model:
    """Load a U‑Net/ResNet model with custom F1 metric."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"f1": f1}
    )


def list_local_models(model_dir="./app_frontend_local/models"):
    models = []
    for fname in os.listdir(model_dir):
        if fname.endswith(".h5"):
            model_id = fname.replace(".h5", "")
            models.append({
                "id": model_id,
                "version": "local",
                "description": f"Local model: {model_id}"
            })
    return models

def get_model_path_local(model_id: str) -> str:
    path = f"./app_frontend_local/models/{model_id}.h5"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{model_id}' not found locally.")
    return path

def list_azure_models():
    ws = Workspace.from_config(path="azureml_config.json")  # Adjust if needed
    models = Model.list(ws)
    return [
        {
            "id": m.name,
            "version": m.version,
            "description": m.description or "",
        }
        for m in models
    ]

# Placeholder for Azure cloud model call
def call_azure_endpoint(model_id: str, image_bytes: bytes):
    print(f"[Mock Azure] Called with model_id: {model_id}, bytes: {len(image_bytes)}")
    
    # Fake output – returning static mock mask shape
    return {
        "mock": True,
        "message": f"Azure model '{model_id}' was called (not really connected).",
        "mask_shape": [512, 512]
    }


# ─── U-Net Model ─────────────────────────────
def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1])
    return model

# ─── Data Generator Loader ─────────────────────────────
def get_generators(patch_dir, patch_size=256, batch_size=32):
    train_img_gen = ImageDataGenerator(rescale=1./255)
    train_mask_gen = ImageDataGenerator()

    val_img_gen = ImageDataGenerator(rescale=1./255)
    val_mask_gen = ImageDataGenerator()

    train_images = train_img_gen.flow_from_directory(
        f"{patch_dir}/images/train", target_size=(patch_size, patch_size),
        batch_size=batch_size, class_mode=None, color_mode='grayscale', seed=42)

    train_masks = train_mask_gen.flow_from_directory(
        f"{patch_dir}/masks/train", target_size=(patch_size, patch_size),
        batch_size=batch_size, class_mode=None, color_mode='grayscale', seed=42)

    val_images = val_img_gen.flow_from_directory(
        f"{patch_dir}/images/test", target_size=(patch_size, patch_size),
        batch_size=batch_size, class_mode=None, color_mode='grayscale', seed=42)

    val_masks = val_mask_gen.flow_from_directory(
        f"{patch_dir}/masks/test", target_size=(patch_size, patch_size),
        batch_size=batch_size, class_mode=None, color_mode='grayscale', seed=42)

    return zip(train_images, train_masks), zip(val_images, val_masks), len(train_images), len(val_images)

# ─── Callback Setup ─────────────────────────────
def get_early_stopping():
    return EarlyStopping(monitor='val_f1', patience=5, restore_best_weights=True, mode='max', verbose=1)
