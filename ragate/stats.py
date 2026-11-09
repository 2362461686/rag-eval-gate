"""Paired significance tests and classifier statistics, on numpy alone."""
import numpy as np

_TIE_TOLERANCE = 1e-12


def sign_flip_p(deltas, n_resamples=10000, seed=0):
    """One-sided paired randomization test of H1: mean delta < 0."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        raise ValueError("no paired observations")
    rng = np.random.default_rng(seed)
    signs = 2.0 * rng.integers(0, 2, size=(n_resamples, deltas.size)) - 1.0
    null = (signs * deltas).mean(axis=1)
    extreme = np.count_nonzero(null <= deltas.mean() + _TIE_TOLERANCE)
    return float((1 + extreme) / (n_resamples + 1))


def bootstrap_ci(deltas, confidence=0.95, n_resamples=10000, seed=0):
    """Percentile bootstrap interval for the mean of the paired deltas."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        raise ValueError("no paired observations")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, deltas.size, size=(n_resamples, deltas.size))
    means = deltas[samples].mean(axis=1)
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def holm(p_values):
    """Holm step-down adjustment, returned in the input order."""
    total = len(p_values)
    adjusted = [None] * total
    running = 0.0
    for position, index in enumerate(np.argsort(p_values, kind="stable")):
        running = max(running, min((total - position) * p_values[index], 1.0))
        adjusted[index] = running
    return adjusted


def roc_auc(labels, scores):
    """Rank-based AUC; tied scores share their average rank."""
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes")
    _, inverse, counts = np.unique(np.asarray(scores, dtype=float), return_inverse=True, return_counts=True)
    ranks = (counts.cumsum() - (counts - 1) / 2)[inverse]
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def binary_report(labels, scores, threshold):
    """Precision, recall and F1 when scores at or above the threshold are positive."""
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores, dtype=float) >= threshold
    true_positive = int((labels & predicted).sum())
    precision = true_positive / max(int(predicted.sum()), 1)
    recall = true_positive / max(int(labels.sum()), 1)
    denominator = int(predicted.sum()) + int(labels.sum())
    f1 = 2 * true_positive / denominator if denominator else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def best_f1_threshold(labels, scores):
    """The score threshold that maximizes F1 on the given labels."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if not labels.any():
        raise ValueError("calibration needs positive examples")
    order = np.argsort(-scores, kind="stable")
    ranked_labels, ranked_scores = labels[order], scores[order]
    true_positive = np.cumsum(ranked_labels)
    predicted = np.arange(1, labels.size + 1)
    proportional_to_f1 = true_positive / (predicted + labels.sum())
    end_of_tie = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    candidates = np.flatnonzero(end_of_tie)
    return float(ranked_scores[candidates[np.argmax(proportional_to_f1[candidates])]])


def bootstrap_statistic_ci(labels, scores, statistic, confidence=0.95, n_resamples=2000, seed=0):
    """Percentile interval of statistic(labels, scores) over resampled examples."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < n_resamples:
        sample = rng.integers(0, labels.size, size=labels.size)
        if labels[sample].all() or not labels[sample].any():
            continue
        values.append(statistic(labels[sample], scores[sample]))
    tail = (1 - confidence) / 2
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))
