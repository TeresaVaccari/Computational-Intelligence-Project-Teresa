from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.ndimage import generic_filter, maximum_filter, minimum_filter
from scipy.io import savemat
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

from wmh_preprocessing import minmax_normalize


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


def parse_args():
    parser = ArgumentParser(
        description="Export preprocessed WMH features to a MATLAB .mat file."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the WMH .npy datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("matlab_preprocessed_features"),
        help="Directory where the .mat file will be saved.",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=31,
        help="Axial slice index to export.",
    )
    return parser.parse_args()


def load_array_or_raise(path):
    try:
        return np.load(path)
    except (EOFError, ValueError, OSError) as exc:
        raise SystemExit(
            f"Could not load {path.resolve()}.\n"
            "Check that the file is present and fully downloaded."
        ) from exc


def validate_slice_idx(slice_idx, volume_shape):
    if not 0 <= slice_idx < volume_shape[2]:
        raise SystemExit(
            f"slice_idx must be between 0 and {volume_shape[2] - 1}; "
            f"received {slice_idx}."
        )


def compute_slice_statistics(flair_slice, window_size=4):
    pad_width = window_size // 2
    padded_slice = np.pad(flair_slice, pad_width=pad_width, mode="reflect")

    mean = generic_filter(padded_slice, np.mean, size=window_size)
    std = generic_filter(padded_slice, np.std, size=window_size)
    skew = generic_filter(padded_slice, scipy_skew, size=window_size)
    kurtosis = generic_filter(padded_slice, scipy_kurtosis, size=window_size)

    crop = slice(pad_width, -pad_width)
    return {
        "mean": mean[crop, crop],
        "std": std[crop, crop],
        "skew": np.nan_to_num(skew[crop, crop], nan=-5.0),
        "kurtosis": np.nan_to_num(kurtosis[crop, crop], nan=-5.0),
    }


def load_precomputed_statistics(data_dir, slice_idx):
    names = {
        "mean": "mean_dataset.npy",
        "std": "std_dataset.npy",
        "skew": "skew_dataset.npy",
        "kurtosis": "kurtosis_dataset.npy",
    }

    statistics = {}
    for name, filename in names.items():
        array = np.load(data_dir / filename)
        statistic_slice = array[:, :, slice_idx]
        if name in {"skew", "kurtosis"}:
            statistic_slice = np.nan_to_num(statistic_slice, nan=-5.0)
        statistics[name] = statistic_slice

    return statistics


def prepare_slice_features_for_export(flair_slice, statistics):
    local_mean = statistics["mean"]
    height, width = flair_slice.shape
    rows = np.repeat(np.arange(height)[:, None], width, axis=1) / height
    cols = np.repeat(np.arange(width)[None, :], height, axis=0) / width

    return {
        "intensity": minmax_normalize(flair_slice),
        "mean": minmax_normalize(local_mean),
        "std": minmax_normalize(statistics["std"]),
        "skew": minmax_normalize(statistics["skew"]),
        "kurtosis": minmax_normalize(statistics["kurtosis"]),
        "contrast": minmax_normalize(flair_slice - local_mean),
        "range": minmax_normalize(
            maximum_filter(flair_slice, size=5) - minimum_filter(flair_slice, size=5)
        ),
        "x": cols,
        "y": rows,
    }


def build_matlab_payload(data_dir, slice_idx):
    flair = load_array_or_raise(data_dir / "FLAIR_dataset.npy")
    masks = load_array_or_raise(data_dir / "WMH_masks.npy")
    validate_slice_idx(slice_idx, flair.shape)

    flair_slice = flair[:, :, slice_idx]
    mask = masks[:, :, slice_idx].astype(np.uint8)

    try:
        statistics = load_precomputed_statistics(data_dir, slice_idx)
    except (EOFError, ValueError, OSError):
        print(
            "Precomputed local-statistic files are not readable; "
            f"recomputing statistics for slice {slice_idx}."
        )
        statistics = compute_slice_statistics(flair_slice)

    features = prepare_slice_features_for_export(flair_slice, statistics)

    feature_matrix = np.column_stack(
        [features[name].reshape(-1) for name in FEATURE_ORDER]
    ).astype(np.float64)

    feature_maps = {
        f"{name}_map": features[name].astype(np.float64)
        for name in FEATURE_ORDER
    }

    return {
        "slice_idx": np.array([[slice_idx]], dtype=np.int32),
        "feature_names": np.array(FEATURE_ORDER, dtype=object),
        "feature_matrix": feature_matrix,
        "manual_mask": mask,
        "manual_mask_vector": mask.reshape(-1, 1),
        "height": np.array([[mask.shape[0]]], dtype=np.int32),
        "width": np.array([[mask.shape[1]]], dtype=np.int32),
        **feature_maps,
    }


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"preprocessed_features_slice_{args.slice_idx}.mat"
    )

    payload = build_matlab_payload(args.data_dir, args.slice_idx)
    savemat(output_path, payload, do_compression=True)

    print(f"Saved {output_path}")
    print(
        "feature_matrix shape: "
        f"{payload['feature_matrix'].shape[0]} pixels x "
        f"{payload['feature_matrix'].shape[1]} features"
    )
    print("Feature order: " + ", ".join(FEATURE_ORDER))


if __name__ == "__main__":
    main()
