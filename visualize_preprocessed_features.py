from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from select_slice import select_slice
from wmh_preprocessing import load_dataset, prepare_slice_features


DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures")
SLICE_IDX = None


def plot_flair_mask_overlay(features, mask, slice_idx):
    plt.figure(figsize=(6, 6))
    plt.imshow(features["intensity"].T, cmap="gray", origin="lower")
    plt.imshow(
        np.ma.masked_where(~mask.T, mask.T),
        cmap="Reds",
        alpha=0.45,
        origin="lower",
    )
    plt.title(f"FLAIR + Manual WMH Mask - Slice {slice_idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"flair_manual_mask_overlay_slice_{slice_idx}.png", dpi=150)
    plt.close()


def plot_feature_maps(features, slice_idx):
    feature_display_names = [
        ("intensity", "Intensity"),
        ("mean", "Mean"),
        ("std", "Std"),
        ("skew", "Skewness"),
        ("kurtosis", "Kurtosis"),
        ("contrast", "Contrast"),
        ("range", "Range"),
        ("x", "X coordinate"),
        ("y", "Y coordinate"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, (feature_name, title) in zip(axes.ravel(), feature_display_names):
        image = ax.imshow(features[feature_name].T, cmap="viridis", origin="lower")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Feature Maps - Slice {slice_idx}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"feature_maps_slice_{slice_idx}.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_dataset(DATA_DIR)
    slice_idx = select_slice() if SLICE_IDX is None else SLICE_IDX

    features = prepare_slice_features(data, slice_idx)
    mask = data["mask"][:, :, slice_idx] == 1

    plot_flair_mask_overlay(features, mask, slice_idx)
    plot_feature_maps(features, slice_idx)


if __name__ == "__main__":
    main()
