import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from app_frontend_final.utils.model_utils import build_unet, build_attention_unet, get_generators, get_early_stopping, f1
import tensorflow as tf

# ─── Config ─────────────────────────────────────────────
patch_size = 256
epochs = 30
model_types = ["unet", "attention_unet"]
base_dataset = Path("./data/original_dataset")
feedback_root = Path("./data/feedback_patches")
output_dir = Path("./data/models")
run_log_dir = Path("./data/training_runs")
dashboard_log_path = Path("./data/dashboard_metrics.csv")

# ─── Collect Feedback Versions ──────────────────────────
feedback_dirs = sorted([
    p / "train" for p in feedback_root.glob("version_*") if (p / "train").exists()
])
all_data_dirs = [base_dataset] + feedback_dirs
feedback_versions = [p.parent.name for p in feedback_dirs]

# ─── Timestamp Setup ────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_id = f"run_{timestamp}"
run_path = run_log_dir / run_id
run_path.mkdir(parents=True, exist_ok=True)

# ─── Train Each Model Type with HParam Tuning ───────────
for model_type in model_types:
    print(f"\n Hyperparameter tuning: {model_type}")
    best_f1 = -1
    best_model = None
    best_config = {}

    for batch_size in [16, 32]:
        for lr in [1e-3, 1e-4]:
            print(f" → Trying batch_size={batch_size}, lr={lr}")
            train_gen, val_gen, steps_per_epoch, val_steps = get_generators(
                all_data_dirs, patch_size, batch_size
            )

            # Build model
            if model_type == "unet":
                model = build_unet((patch_size, patch_size, 1))
            elif model_type == "attention_unet":
                model = build_attention_unet((patch_size, patch_size, 1))
            else:
                raise ValueError("Unsupported model_type")

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="binary_crossentropy",
                metrics=["accuracy", f1]
            )

            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=val_steps,
                callbacks=[get_early_stopping()],
                verbose=0
            )

            val_f1 = max(history.history.get("val_f1", [0]))
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = model
                best_history = history
                best_config = {
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "val_f1": val_f1
                }

    # Load previous model's performance (if exists)
    registry_path = Path("./app_frontend_local/models/registry.json")
    previous_f1 = -1
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
            previous_versions = [r for r in registry if r["model_type"] == model_type]
            if previous_versions:
                previous_f1 = max(r.get("val_f1", 0) for r in previous_versions)

    # Compare & save if improved
    if best_f1 > previous_f1:
        model_name = f"{model_type}_{timestamp}_fb_versions_{'-'.join(feedback_versions)}.h5"
        model_path = output_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model.save(model_path)
        print(f"Registered new model: {model_path}")

        # Save logs
        run_meta = {
            "model_type": model_type,
            "timestamp": timestamp,
            "model_path": str(model_path),
            "base_dataset": str(base_dataset),
            "feedback_versions": feedback_versions,
            "params": {
                "patch_size": patch_size,
                "epochs": epochs,
                **best_config
            },
            "metrics": {
                "val_f1": best_f1
            }
        }
        with open(run_path / f"{model_type}_config.json", "w") as f:
            json.dump(run_meta, f, indent=2)

        plt.figure()
        plt.plot(best_history.history['val_loss'], label='val_loss')
        plt.plot(best_history.history.get('val_f1', []), label='val_f1')
        plt.title(f"{model_type} Learning Curve")
        plt.legend()
        plt.savefig(run_path / f"{model_type}_learning_curve.png")
        plt.close()

        # Update registry
        if registry_path.exists():
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            registry = []
        registry.append(run_meta)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        # Append to dashboard CSV
        row = {
            "model_type": model_type,
            "timestamp": timestamp,
            "val_f1": best_f1,
            "batch_size": best_config["batch_size"],
            "learning_rate": best_config["learning_rate"],
            "feedback_versions": '|'.join(feedback_versions),
        }
        if dashboard_log_path.exists():
            df = pd.read_csv(dashboard_log_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(dashboard_log_path, index=False)

    else:
        print(f"Skipped saving {model_type}: not better than previous (val_f1={previous_f1})")

print("\nTraining & evaluation complete for all model types.")
