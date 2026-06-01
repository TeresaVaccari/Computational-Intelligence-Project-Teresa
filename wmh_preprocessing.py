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

# use MinMaxScaler to normalize features to [0, 1]
def minmax_normalize(x):
    original_shape = x.shape # shape originale (151, 186, 611)
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_flat = x.reshape(-1, 1)
    x_norm = scaler.fit_transform(x_flat)
    return x_norm.reshape(original_shape)

# prepare normalized pixel features for one 2D slice.
def prepare_slice_features(data, slice_idx):
    # each returned feature is a 2D array with the same shape as the selected slice
    flair = data["flair"][:, :, slice_idx]

    # local mean: mean FLAIR intensity in a local neighborhood around each
    mean = minmax_normalize(data["mean"][:, :, slice_idx])

    # local standard deviation: variability of intensities in the local neighborhood 
    std = minmax_normalize(data["std"][:, :, slice_idx])

    skew = minmax_normalize(data["skew"][:, :, slice_idx])

    kurtosis = minmax_normalize(data["kurtosis"][:, :, slice_idx])

    # pixel intensity: original FLAIR value of each pixel, normalized to [0, 1]
    intensity = minmax_normalize(flair)

    # local contrast: difference between the pixel intensity and its local mean
    contrast = minmax_normalize(flair - data["mean"][:, :, slice_idx])

    # local range: maximum - minimum intensity in a 5x5 window around each pixel
    range_raw = maximum_filter(flair, size=5) - minimum_filter(flair, size=5)
    range = minmax_normalize(range_raw)

    # spatial features: normalized pixel coordinates
    H, W = flair.shape
    rows = np.repeat(np.arange(H)[:, None], W, axis=1) / H  # coordinata y
    cols = np.repeat(np.arange(W)[None, :], H, axis=0) / W  # coordinata x


    return {
        
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "intensity": intensity,
        "contrast": contrast,
        "range": range,
        "x": cols,  
        "y": rows,   
    }
