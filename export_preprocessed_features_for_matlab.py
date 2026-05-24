from pathlib import Path

import numpy as np
from scipy.io import savemat

from wmh_preprocessing import load_dataset, prepare_slice_features


PROJECT_ROOT = Path(__file__).resolve().parent


def choose_slice_with_most_wmh(mask_volume):
    # We use the slice with the largest manual WMH area as an example for MATLAB.
    mask_pixels_per_slice = mask_volume.sum(axis=(0, 1))
    return int(np.argmax(mask_pixels_per_slice))


def main():
    data = load_dataset(PROJECT_ROOT / "data")
    slice_idx = choose_slice_with_most_wmh(data["mask"])
    features = prepare_slice_features(data, slice_idx)

    # Feature order must match wmh_fis_rules.m:
    # [intensity local_mean local_std skewness kurtosis local_contrast local_range]
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

    flair_slice = data["flair"][:, :, slice_idx]
    manual_mask = data["mask"][:, :, slice_idx]
    image_shape = np.array(flair_slice.shape)

    output_dir = PROJECT_ROOT / "matlab_preprocessed_features"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"preprocessed_features_slice_{slice_idx + 1}.mat"

    savemat(
        output_path,
        {
            "sliceIdxMatlab": slice_idx + 1,
            "featureMatrix": feature_matrix,
            "manualMask": manual_mask,
            "flairSlice": flair_slice,
            "imageShape": image_shape,
        },
    )

    print(f"Saved MATLAB features to: {output_path}")
    print(f"MATLAB slice index: {slice_idx + 1}")
    print(f"Feature matrix shape: {feature_matrix.shape}")


if __name__ == "__main__":
    main()
