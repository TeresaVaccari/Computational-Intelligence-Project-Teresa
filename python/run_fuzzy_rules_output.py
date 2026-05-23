from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wmh_fis_python_rules import N_RULES, apply_threshold, fis_score
from wmh_preprocessing import load_dataset, prepare_slice_features


def dice_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denom = prediction.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def main():
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "fuzzy_rules_output"
    output_dir.mkdir(exist_ok=True)

    data = load_dataset(data_dir)

    mask_pixels = data["mask"].reshape(-1, data["mask"].shape[-1]).sum(axis=0)
    slice_idx = int(np.argmax(mask_pixels))

    features = prepare_slice_features(data, slice_idx)

    # Default MATLAB FIS rule weights are 1.
    rule_weights = np.ones(N_RULES)
    score = fis_score(features, rule_weights)
    prediction = apply_threshold(score, 0.5)
    target = data["mask"][:, :, slice_idx] > 0
    dice = dice_score(prediction, target)

    np.save(output_dir / "fis_score.npy", score)
    np.save(output_dir / "fis_prediction_threshold_0_5.npy", prediction)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Fuzzy rules output generated with Python replica of MATLAB FIS.\n")
        f.write(f"Slice index: {slice_idx}\n")
        f.write("Rule weights: all ones, same as default MATLAB ruleList weights.\n")
        f.write("Threshold: 0.5\n")
        f.write(f"Score shape: {score.shape}\n")
        f.write(f"Score min: {score.min():.6f}\n")
        f.write(f"Score max: {score.max():.6f}\n")
        f.write(f"Score mean: {score.mean():.6f}\n")
        f.write(f"Dice vs manual mask: {dice:.6f}\n")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(data["flair"][:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")
    axes[1].imshow(target.T, cmap="gray", origin="lower")
    axes[1].set_title("Manual mask")
    axes[2].imshow(score.T, cmap="magma", origin="lower")
    axes[2].set_title("FIS score")
    axes[3].imshow(prediction.T, cmap="gray", origin="lower")
    axes[3].set_title(f"Threshold 0.5 | Dice={dice:.3f}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "fis_output_visualization.png", dpi=150)
    plt.close(fig)

    print(f"Saved fuzzy output in: {output_dir}")
    print(f"Slice index: {slice_idx}")
    print(f"Dice vs manual mask: {dice:.4f}")


if __name__ == "__main__":
    main()
