import numpy as np


EPS = 1e-8

# Same rule order used in wmh_fis_matlab_rules.m.
# +1 means the rule supports WMH, -1 means it supports non-WMH.
RULE_CONSEQUENTS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1], dtype=float)
N_RULES = len(RULE_CONSEQUENTS)


def trapezoid(x, a, b, c, d):
    # Python equivalent of MATLAB trapmf.
    y = np.zeros_like(x, dtype=float)

    if a == b:
        y[(x >= a) & (x <= c)] = 1.0
    else:
        rising = (x > a) & (x < b)
        y[rising] = (x[rising] - a) / (b - a + EPS)
        y[(x >= b) & (x <= c)] = 1.0

    if c == d:
        y[(x >= b) & (x <= d)] = 1.0
    else:
        falling = (x > c) & (x < d)
        y[falling] = (d - x[falling]) / (d - c + EPS)

    return np.clip(y, 0.0, 1.0)


def triangle(x, a, b, c):
    # Python equivalent of MATLAB trimf.
    left = (x - a) / (b - a + EPS)
    right = (c - x) / (c - b + EPS)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def fuzzy_rule_activations(features):
    # Inputs follow the same order/concepts as the MATLAB FIS:
    # intensity, local_mean, local_std, skewness, kurtosis, local_contrast, local_range.
    intensity = features["intensity"]
    local_mean = features["mean"]
    local_std = features["std"]
    skew = features["skew"]
    kurtosis = features["kurtosis"]
    contrast = features["contrast"]
    local_range = features["range"]

    intensity_low = trapezoid(intensity, 0.0, 0.0, 0.20, 0.45)
    intensity_medium = triangle(intensity, 0.25, 0.50, 0.75)
    intensity_high = trapezoid(intensity, 0.55, 0.75, 1.0, 1.0)

    mean_low = trapezoid(local_mean, 0.0, 0.0, 0.20, 0.45)
    mean_high = trapezoid(local_mean, 0.55, 0.75, 1.0, 1.0)

    std_low = trapezoid(local_std, 0.0, 0.0, 0.20, 0.45)
    std_medium = triangle(local_std, 0.25, 0.50, 0.75)
    std_high = trapezoid(local_std, 0.55, 0.75, 1.0, 1.0)

    contrast_low = trapezoid(contrast, 0.0, 0.0, 0.20, 0.45)
    contrast_high = trapezoid(contrast, 0.55, 0.75, 1.0, 1.0)

    range_low = trapezoid(local_range, 0.0, 0.0, 0.20, 0.45)
    range_high = trapezoid(local_range, 0.55, 0.75, 1.0, 1.0)

    skew_high = trapezoid(skew, 0.55, 0.75, 1.0, 1.0)
    kurtosis_high = trapezoid(kurtosis, 0.55, 0.75, 1.0, 1.0)

    return np.stack(
        [
            np.minimum(intensity_high, mean_high),                      # R1
            np.minimum(intensity_high, contrast_high),                  # R2
            np.minimum(intensity_high, std_medium),                     # R3
            np.minimum(intensity_high, std_high),                       # R4
            np.minimum(mean_high, kurtosis_high),                       # R5
            np.minimum(intensity_high, skew_high),                      # R6
            np.minimum.reduce([intensity_medium, mean_high, std_high]), # R7
            np.minimum(intensity_high, range_high),                     # R8
            np.minimum.reduce([intensity_medium, mean_high, range_high]), # R9
            intensity_low,                                              # R10
            mean_low,                                                   # R11
            np.minimum.reduce([std_low, contrast_low, range_low]),      # R12
        ],
        axis=0,
    )


def fis_score(features, rule_weights):
    activations = fuzzy_rule_activations(features)
    signed_weights = rule_weights[:, None, None] * RULE_CONSEQUENTS[:, None, None]
    evidence = np.sum(signed_weights * activations, axis=0)
    return 1.0 / (1.0 + np.exp(-4.0 * evidence))


def apply_threshold(score, threshold):
    return score >= threshold
