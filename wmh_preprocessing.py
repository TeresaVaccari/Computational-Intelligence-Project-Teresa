import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.preprocessing import MinMaxScaler


def load_dataset(data_dir):
    return {
        "flair": np.load(data_dir / "FLAIR_dataset.npy"),
        "mask": np.load(data_dir / "WMH_masks.npy"),
        "mean": np.load(data_dir / "mean_dataset.npy"),
        "std": np.load(data_dir / "std_dataset.npy"),
        "skew": np.nan_to_num(np.load(data_dir / "skew_dataset.npy"), nan=-5.0),
        "kurtosis": np.nan_to_num(np.load(data_dir / "kurtosis_dataset.npy"), nan=-5.0),
    }


def minmax_normalize(x):
    # MinMaxScaler maps values into the [0, 1] interval.
    original_shape = x.shape
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_flat = x.reshape(-1, 1)
    x_norm = scaler.fit_transform(x_flat)
    return x_norm.reshape(original_shape)


def prepare_slice_features(data, slice_idx):
    flair = data["flair"][:, :, slice_idx]

    # Precomputed sliding-window features loaded from the provided .npy files.
    local_mean = minmax_normalize(data["mean"][:, :, slice_idx])
    local_std = minmax_normalize(data["std"][:, :, slice_idx])
    skew = minmax_normalize(data["skew"][:, :, slice_idx])
    kurtosis = minmax_normalize(data["kurtosis"][:, :, slice_idx])

    # Pixel intensity feature: normalized original FLAIR intensity.
    intensity = minmax_normalize(flair)

    # Local contrast feature: how much brighter/darker the pixel is than its local mean.
    contrast = minmax_normalize(flair - data["mean"][:, :, slice_idx])

    # Local range feature: max(window) - min(window) using a 5x5 sliding window.
    local_range_raw = maximum_filter(flair, size=5) - minimum_filter(flair, size=5)
    local_range = minmax_normalize(local_range_raw)

    # Feature spaziali: coordinate x,y normalizzate in [0, 1]
    H, W = flair.shape
    rows = np.repeat(np.arange(H)[:, None], W, axis=1) / H  # coordinata y
    cols = np.repeat(np.arange(W)[None, :], H, axis=0) / W  # coordinata x


    return {
        
        "mean": local_mean,
        "std": local_std,
        "skew": skew,
        "kurtosis": kurtosis,
        "intensity": intensity,
        "contrast": contrast,
        "range": local_range,
        "x": cols,  
        "y": rows,   
    }
