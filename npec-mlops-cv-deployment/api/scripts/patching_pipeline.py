# main.py
import argparse
from pathlib import Path

from data_ingestion import get_valid_image_mask_pairs, load_image_file, load_mask_tif
from data_preprocessing import cropper, padder
from data_patchify import patchify_pair
from data_filter import filter_patches_by_mask_content, classify_background_patches
from data_augment import augment_patches, sample_and_merge_background
from data_save_output import save_patch_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--output_images_dir", type=str, required=True)
    parser.add_argument("--output_masks_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--step", type=int, default=128)
    args = parser.parse_args()

    valid_keys = get_valid_image_mask_pairs(args.images_dir, args.masks_dir)

    all_patches = []

    for key in valid_keys:
        image_path = Path(args.images_dir) / f"{key}.png"
        mask_path = Path(args.masks_dir) / f"{key}_root_mask.tif"

        image = load_image_file(image_path)
        mask, _ = load_mask_tif(mask_path)

        cropped_img, cropped_mask, _ = cropper(image, mask)
        padded_img, padded_mask, _ = padder(cropped_img, cropped_mask, args.patch_size)

        patches = patchify_pair(padded_img, padded_mask, args.patch_size, args.step)
        all_patches.extend(patches)

    # Filter and balance
    root_rich, background = filter_patches_by_mask_content(all_patches)
    clean_bg, noisy_bg = classify_background_patches(background)
    augmented_root = augment_patches(root_rich)
    background_sampled = sample_and_merge_background(clean_bg, noisy_bg, 0.15, 0.25, len(augmented_root))

    # Save final dataset
    final_patches = augmented_root + background_sampled
    save_patch_dataset(final_patches, args.output_images_dir, args.output_masks_dir)


if __name__ == "__main__":
    main()
