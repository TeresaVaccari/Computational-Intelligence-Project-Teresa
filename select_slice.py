from pathlib import Path

import numpy as np


DATA_DIR = Path("data")


def select_slice():
    mask = np.load(DATA_DIR / "WMH_masks.npy")
    lesion_counts = mask.sum(axis=(0, 1))

    slice_idx = int(np.argmax(lesion_counts))
    lesion_pixels = int(lesion_counts[slice_idx])

    print(f"Selected slice index: {slice_idx}")
    print(f"WMH lesion pixels: {lesion_pixels}")

    return slice_idx


if __name__ == "__main__":
    select_slice()
