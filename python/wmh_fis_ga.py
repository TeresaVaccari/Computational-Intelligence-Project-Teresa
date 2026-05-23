import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wmh_fis_python_rules import N_RULES, apply_threshold, fis_score
from wmh_preprocessing import load_dataset, prepare_slice_features


def segment_slice(
    data,
    slice_idx,
    rule_weights,
    threshold=0.5,
):
    features = prepare_slice_features(data, slice_idx)
    score = fis_score(features, rule_weights)
    return apply_threshold(score, threshold)


def dice_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denom = prediction.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def select_training_slices(data, max_slices, min_mask_pixels):
    mask_pixels = data["mask"].reshape(-1, data["mask"].shape[-1]).sum(axis=0)
    candidates = np.where(mask_pixels >= min_mask_pixels)[0]
    if len(candidates) == 0:
        candidates = np.argsort(mask_pixels)[-max_slices:]
    return candidates[:max_slices].tolist()


def evaluate_weights(
    data,
    slice_indices,
    rule_weights,
    threshold,
):
    scores = []
    for idx in slice_indices:
        prediction = segment_slice(data, idx, rule_weights, threshold)
        target = data["mask"][:, :, idx] > 0
        scores.append(dice_score(prediction, target))
    return float(np.mean(scores))


def find_best_threshold(
    data,
    slice_indices,
    rule_weights,
    thresholds,
):
    # Tests different thresholds and selects the one with the highest Dice score.
    best_threshold = float(thresholds[0])
    best_dice = -1.0

    for threshold in thresholds:
        dice = evaluate_weights(data, slice_indices, rule_weights, float(threshold))
        if dice > best_dice:
            best_dice = dice
            best_threshold = float(threshold)

    return best_threshold, best_dice


def genetic_algorithm(
    data,
    slice_indices,
    population_size=40,
    generations=30,
    mutation_rate=0.15,
    threshold=0.5,
    seed=42,
):
    rng = np.random.default_rng(seed)
    population = rng.uniform(0.0, 1.0, size=(population_size, N_RULES))
    history = []

    for generation in range(generations):
        fitness = np.array(
            [evaluate_weights(data, slice_indices, individual, threshold) for individual in population]
        )
        order = np.argsort(fitness)[::-1]
        population = population[order]
        fitness = fitness[order]
        history.append(float(fitness[0]))
        print(f"Generation {generation + 1:03d} | best Dice = {fitness[0]:.4f}")

        elite_count = max(2, population_size // 5)
        new_population = [population[i].copy() for i in range(elite_count)]

        while len(new_population) < population_size:
            parent_a = tournament_select(population, fitness, rng)
            parent_b = tournament_select(population, fitness, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, mutation_rate, rng)
            new_population.append(child)

        population = np.array(new_population)

    fitness = np.array(
        [evaluate_weights(data, slice_indices, individual, threshold) for individual in population]
    )
    best_idx = int(np.argmax(fitness))
    return population[best_idx], float(fitness[best_idx]), history


def tournament_select(
    population,
    fitness,
    rng,
    tournament_size=3,
):
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    return population[indices[np.argmax(fitness[indices])]]


def crossover(parent_a, parent_b, rng):
    mask = rng.random(parent_a.shape) < 0.5
    return np.where(mask, parent_a, parent_b)


def mutate(child, mutation_rate, rng):
    mutation_mask = rng.random(child.shape) < mutation_rate
    noise = rng.normal(0.0, 0.12, size=child.shape)
    child = child + mutation_mask * noise
    return np.clip(child, 0.0, 1.0)


def plot_result(
    data,
    slice_idx,
    rule_weights,
    output_path,
    threshold=0.5,
):
    features = prepare_slice_features(data, slice_idx)
    score = fis_score(features, rule_weights)
    prediction = apply_threshold(score, threshold)
    target = data["mask"][:, :, slice_idx] > 0
    dice = dice_score(prediction, target)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(data["flair"][:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")
    axes[1].imshow(target.T, cmap="gray", origin="lower")
    axes[1].set_title("Manual mask")
    axes[2].imshow(score.T, cmap="magma", origin="lower")
    axes[2].set_title("FIS score")
    axes[3].imshow(prediction.T, cmap="gray", origin="lower")
    axes[3].set_title(f"Prediction | Dice={dice:.3f}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="WMH segmentation with FIS and GA-tuned rule weights.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--max-slices", type=int, default=3)
    parser.add_argument("--min-mask-pixels", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("wmh_fis_ga_result.png"))
    args = parser.parse_args()

    data = load_dataset(args.data_dir)
    training_slices = select_training_slices(data, args.max_slices, args.min_mask_pixels)
    print(f"Training slices: {training_slices}")

    baseline_weights = np.ones(N_RULES) * 0.5
    baseline_dice = evaluate_weights(data, training_slices, baseline_weights, args.threshold)
    print(f"Baseline Dice with equal weights: {baseline_dice:.4f}")

    best_weights, best_dice, _ = genetic_algorithm(
        data=data,
        slice_indices=training_slices,
        population_size=args.population_size,
        generations=args.generations,
        threshold=args.threshold,
        seed=args.seed,
    )

    print("\nBest rule weights:")
    for rule_idx, weight in enumerate(best_weights, start=1):
        print(f"Rule {rule_idx:02d}: {weight:.3f}")
    print(f"\nBest Dice: {best_dice:.4f}")

    final_threshold = args.threshold
    if args.tune_threshold:
        threshold_grid = np.linspace(0.1, 0.9, 17)
        final_threshold, threshold_dice = find_best_threshold(
            data, training_slices, best_weights, threshold_grid
        )
        print(f"Best threshold: {final_threshold:.2f}")
        print(f"Dice with best threshold: {threshold_dice:.4f}")

    plot_result(data, training_slices[0], best_weights, args.output, final_threshold)
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
