import itertools

import numpy as np
import pytest

from ragate.stats import (
    best_f1_threshold,
    binary_report,
    bootstrap_ci,
    bootstrap_statistic_ci,
    holm,
    roc_auc,
    sign_flip_p,
)


def _exact_sign_flip(deltas):
    deltas = np.array(deltas, dtype=float)
    assignments = np.array(list(itertools.product((-1.0, 1.0), repeat=deltas.size)))
    return float(np.mean((assignments * deltas).mean(axis=1) <= deltas.mean() + 1e-12))


@pytest.mark.parametrize(
    "deltas",
    [[-0.2, -0.1, 0.05, -0.3, 0.02, -0.07, -0.01, 0.1], [0.3, 0.1, -0.05, 0.2, 0.2, -0.02, 0.01, 0.04]],
)
def test_sign_flip_matches_exact_enumeration(deltas):
    assert sign_flip_p(deltas, n_resamples=60000) == pytest.approx(_exact_sign_flip(deltas), abs=0.01)


def test_sign_flip_is_one_when_nothing_changed():
    assert sign_flip_p(np.zeros(30), n_resamples=500) == 1.0


def test_sign_flip_is_tiny_for_a_uniform_drop():
    assert sign_flip_p(-np.ones(30), n_resamples=999) == pytest.approx(1 / 1000)


def test_sign_flip_is_large_for_an_improvement():
    assert sign_flip_p(np.ones(30), n_resamples=999) > 0.99


def test_sign_flip_is_reproducible_and_seed_sensitive():
    deltas = np.random.default_rng(1).normal(-0.01, 0.1, 40)
    assert sign_flip_p(deltas, n_resamples=300, seed=5) == sign_flip_p(deltas, n_resamples=300, seed=5)
    assert sign_flip_p(deltas, n_resamples=300, seed=5) != sign_flip_p(deltas, n_resamples=300, seed=6)


def test_sign_flip_holds_its_false_positive_rate():
    rng = np.random.default_rng(7)
    p_values = [sign_flip_p(rng.normal(0, 0.1, 60), n_resamples=400, seed=i) for i in range(400)]
    assert 0.02 <= np.mean(np.array(p_values) < 0.05) <= 0.09


@pytest.mark.parametrize("function", [sign_flip_p, bootstrap_ci])
def test_paired_statistics_reject_empty_input(function):
    with pytest.raises(ValueError):
        function([])


def test_bootstrap_ci_collapses_on_constant_deltas():
    assert bootstrap_ci(np.full(20, -0.25), n_resamples=200) == pytest.approx((-0.25, -0.25))


def test_bootstrap_ci_matches_the_normal_approximation():
    deltas = np.random.default_rng(3).normal(0.5, 1.0, 400)
    half_width = 1.96 * deltas.std(ddof=1) / np.sqrt(deltas.size)
    low, high = bootstrap_ci(deltas, n_resamples=20000)
    assert low == pytest.approx(deltas.mean() - half_width, abs=0.005)
    assert high == pytest.approx(deltas.mean() + half_width, abs=0.005)


def test_bootstrap_ci_resamples_every_observation():
    deltas = np.zeros(10)
    deltas[0] = 100.0
    assert bootstrap_ci(deltas, n_resamples=2000)[1] > 0


def test_bootstrap_ci_widens_with_confidence():
    deltas = np.random.default_rng(3).normal(0, 1, 100)
    narrow = bootstrap_ci(deltas, confidence=0.5, n_resamples=4000)
    wide = bootstrap_ci(deltas, confidence=0.99, n_resamples=4000)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]


def test_holm_matches_hand_computed_values_in_input_order():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_capped_at_one():
    assert holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_holm_keeps_a_step_down_monotone():
    assert holm([0.001, 0.2, 0.021]) == pytest.approx([0.003, 0.2, 0.042])


def test_roc_auc_on_a_hand_computed_example():
    assert roc_auc([1, 1, 0, 0], [0.9, 0.2, 0.5, 0.1]) == pytest.approx(0.75)


def test_roc_auc_extremes():
    assert roc_auc([1, 0], [1.0, 0.0]) == 1.0
    assert roc_auc([1, 0], [0.0, 1.0]) == 0.0


def test_roc_auc_counts_ties_as_half():
    assert roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert roc_auc([1, 0, 0], [0.5, 0.5, 0.1]) == pytest.approx(0.75)


@pytest.mark.parametrize("labels", [[1, 1], [0, 0]])
def test_roc_auc_needs_both_classes(labels):
    with pytest.raises(ValueError):
        roc_auc(labels, [0.1, 0.2])


def test_binary_report_on_a_hand_computed_example():
    report = binary_report([1, 1, 0, 0, 1], [0.9, 0.2, 0.8, 0.1, 0.7], 0.5)
    assert report == pytest.approx({"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3})


def test_binary_report_counts_the_threshold_as_positive():
    assert binary_report([1], [0.5], 0.5)["recall"] == 1.0


def test_binary_report_with_differing_precision_and_recall():
    report = binary_report([1, 1, 1, 0], [0.9, 0.1, 0.1, 0.1], 0.5)
    assert report == pytest.approx({"precision": 1.0, "recall": 1 / 3, "f1": 0.5})


def test_binary_report_when_nothing_is_predicted_or_labelled():
    assert binary_report([0, 0], [0.1, 0.2], 0.5) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_best_f1_threshold_on_a_hand_computed_example():
    assert best_f1_threshold([1, 1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.1]) == pytest.approx(0.6)


def test_best_f1_threshold_prefers_precision_when_it_pays():
    assert best_f1_threshold([1, 1, 0, 0, 0, 0], [0.9, 0.8, 0.3, 0.3, 0.3, 0.3]) == pytest.approx(0.8)


def test_best_f1_threshold_never_splits_tied_scores():
    assert best_f1_threshold([1, 0, 0, 0], [0.7, 0.7, 0.7, 0.2]) == pytest.approx(0.7)
    assert best_f1_threshold([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_best_f1_threshold_scores_a_whole_tie_group_together():
    assert best_f1_threshold([1, 1, 0, 0, 0], [0.9, 0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.9)


def test_best_f1_threshold_needs_positive_examples():
    with pytest.raises(ValueError):
        best_f1_threshold([0, 0], [0.1, 0.2])


def test_bootstrap_statistic_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 400).astype(bool)
    scores = labels * 0.3 + rng.random(400)
    point = roc_auc(labels, scores)
    low, high = bootstrap_statistic_ci(labels, scores, roc_auc, n_resamples=400)
    assert low < point < high
    assert high - low < 0.2


def test_bootstrap_statistic_ci_skips_single_class_resamples():
    low, high = bootstrap_statistic_ci([1, 0], [1.0, 0.0], roc_auc, n_resamples=20)
    assert (low, high) == (1.0, 1.0)


def test_bootstrap_statistic_ci_uses_exactly_the_requested_tails_and_resamples():
    counter = itertools.count()
    interval = bootstrap_statistic_ci([1, 0], [1.0, 0.0], lambda *_: float(next(counter)), confidence=0.8, n_resamples=101)
    assert interval == pytest.approx((10.0, 90.0))


def test_bootstrap_statistic_ci_widens_with_confidence():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, 300).astype(bool)
    scores = labels * 0.2 + rng.random(300)
    narrow = bootstrap_statistic_ci(labels, scores, roc_auc, confidence=0.5, n_resamples=500)
    wide = bootstrap_statistic_ci(labels, scores, roc_auc, confidence=0.99, n_resamples=500)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]
