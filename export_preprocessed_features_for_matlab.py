from pathlib import Path

import numpy as np
from scipy.io import savemat

from wmh_preprocessing import load_dataset, prepare_slice_features


DATA_DIR = Path("data")
OUTPUT_DIR = Path("matlab")
SLICE_IDX = 31

FEATURE_ORDER = [
    "intensity",
    "mean",
    "std",
    "skew",
    "kurtosis",
    "contrast",
    "range",
    "x",
    "y",
]


def main():
    data = load_dataset(DATA_DIR)

    features = prepare_slice_features(data, SLICE_IDX)
    mask = data["mask"][:, :, SLICE_IDX].astype(np.uint8)

    feature_matrix = np.column_stack(
        [features[name].reshape(-1) for name in FEATURE_ORDER]
    )

    payload = {
        "feature_names": np.array(FEATURE_ORDER, dtype=object),
        "feature_matrix": feature_matrix,
        "manual_mask": mask,
        "manual_mask_vector": mask.reshape(-1, 1),
        "intensity_map": features["intensity"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"preprocessed_features_slice_{SLICE_IDX}.mat"
    savemat(output_path, payload, do_compression=True)

    print(f"Saved {output_path}")
    print(f"feature_matrix: {feature_matrix.shape[0]} pixels x {feature_matrix.shape[1]} features")
    print("Feature order: " + ", ".join(FEATURE_ORDER))


if __name__ == "__main__":
    main()

