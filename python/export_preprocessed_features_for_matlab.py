from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wmh_preprocessing import load_dataset, prepare_slice_features


def choose_slice_with_wmh(mask):
    mask_pixels = mask.reshape(-1, mask.shape[-1]).sum(axis=0)
    return int(np.argmax(mask_pixels))


def main():
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "matlab_preprocessed_features"
    output_dir.mkdir(exist_ok=True)

    data = load_dataset(data_dir)
    slice_idx = choose_slice_with_wmh(data["mask"])
    features = prepare_slice_features(data, slice_idx)

    # MATLAB evalfis expects one row per pixel and one column per feature.
    # Feature order must match wmh_build_fis.m:
    # intensity, local_mean, local_std, skewness, kurtosis, local_contrast, local_range.
    feature_matrix = np.column_stack(
        [
            features["intensity"].ravel(),
            features["mean"].ravel(),
            features["std"].ravel(),
            features["skew"].ravel(),
            features["kurtosis"].ravel(),
            features["contrast"].ravel(),
            features["range"].ravel(),
        ]
    )

    output_path = output_dir / f"preprocessed_features_slice_{slice_idx}.mat"
    savemat(
        output_path,
        {
            "featureMatrix": feature_matrix,
            "sliceIdxPython": slice_idx,
            "sliceIdxMatlab": slice_idx + 1,
            "imageShape": np.array(data["flair"][:, :, slice_idx].shape),
            "flairSlice": data["flair"][:, :, slice_idx],
            "manualMask": data["mask"][:, :, slice_idx],
            "featureNames": np.array(
                [
                    "intensity",
                    "local_mean",
                    "local_std",
                    "skewness",
                    "kurtosis",
                    "local_contrast",
                    "local_range",
                ],
                dtype=object,
            ),
        },
    )

    print(f"Saved MATLAB preprocessing input: {output_path}")
    print(f"Python slice index: {slice_idx}")
    print(f"MATLAB slice index: {slice_idx + 1}")
    print(f"featureMatrix shape: {feature_matrix.shape}")


if __name__ == "__main__":
    main()
