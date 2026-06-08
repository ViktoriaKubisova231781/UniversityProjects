import argparse
import shutil
from pathlib import Path
from azureml_deployment.preprocess_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="AzureML entry point for image-mask preprocessing")

    # Inputs
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)

    # Output paths
    parser.add_argument("--train_output_dir", type=str, required=True)
    parser.add_argument("--val_output_dir", type=str, required=True)
    parser.add_argument("--test_output_dir", type=str, required=True)

    # Parameters
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--step_size", type=int, default=128)
    parser.add_argument("--mask_threshold", type=int, default=150)
    parser.add_argument("--clean_ratio", type=float, default=0.6)
    parser.add_argument("--noisy_ratio", type=float, default=0.4)
    parser.add_argument("--split_ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1])

    args = parser.parse_args()

    # Local temp output root
    temp_output_root = Path("outputs")
    temp_output_root.mkdir(parents=True, exist_ok=True)

    # Wrap arguments
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    pipeline_args = Struct(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=str(temp_output_root),
        patch_size=args.patch_size,
        step_size=args.step_size,
        mask_threshold=args.mask_threshold,
        clean_ratio=args.clean_ratio,
        noisy_ratio=args.noisy_ratio,
        split_ratio=args.split_ratio,
    )

    run_pipeline(pipeline_args)

    # Move to AzureML-designated output folders
    shutil.move(str(temp_output_root / "train"), args.train_output_dir)
    shutil.move(str(temp_output_root / "val"), args.val_output_dir)
    shutil.move(str(temp_output_root / "test"), args.test_output_dir)

    print("Patchified data saved to AzureML output folders.")

if __name__ == "__main__":
    main()
