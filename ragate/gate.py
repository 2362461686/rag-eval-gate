"""Decide whether a candidate retrieval run regressed against a baseline run.

A metric fails only when the drop is both statistically significant (paired
sign-flip test, Holm-adjusted across metrics) and at least `margin` in size.
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragate.stats import bootstrap_ci, holm, sign_flip_p


@dataclass(frozen=True)
class Verdict:
    metric: str
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    regressed: bool


def _paired(baseline, candidate):
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("baseline and candidate must report the same metrics")
    pairs = {}
    for name in sorted(baseline):
        if set(baseline[name]) != set(candidate[name]):
            raise ValueError(f"{name}: baseline and candidate cover different queries")
        queries = sorted(baseline[name])
        pairs[name] = (
            np.array([baseline[name][query] for query in queries], dtype=float),
            np.array([candidate[name][query] for query in queries], dtype=float),
        )
    return pairs


def compare(baseline, candidate, margin=0.01, alpha=0.05, n_resamples=10000, seed=0):
    """Return one Verdict per metric for two {metric: {query: value}} mappings."""
    pairs = _paired(baseline, candidate)
    names = list(pairs)
    deltas = {name: pairs[name][1] - pairs[name][0] for name in names}
    p_values = [sign_flip_p(deltas[name], n_resamples, seed) for name in names]
    verdicts = []
    for name, p_value, adjusted in zip(names, p_values, holm(p_values)):
        low, high = bootstrap_ci(deltas[name], n_resamples=n_resamples, seed=seed)
        delta = float(deltas[name].mean())
        verdicts.append(
            Verdict(
                metric=name,
                baseline=float(pairs[name][0].mean()),
                candidate=float(pairs[name][1].mean()),
                delta=delta,
                ci_low=low,
                ci_high=high,
                p_value=p_value,
                p_adjusted=adjusted,
                regressed=adjusted < alpha and delta <= -margin,
            )
        )
    return verdicts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["metrics"]
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))["metrics"]
        verdicts = compare(baseline, candidate, margin=args.margin, alpha=args.alpha)
    except (OSError, KeyError, ValueError) as error:
        print(f"invalid input: {error}", file=sys.stderr)
        return 2
    for v in verdicts:
        state = "REGRESSED" if v.regressed else "ok"
        print(
            f"{state:9} {v.metric:10} {v.baseline:.4f} -> {v.candidate:.4f}"
            f"  delta {v.delta:+.4f}  95% CI [{v.ci_low:+.4f}, {v.ci_high:+.4f}]  p_holm {v.p_adjusted:.4f}"
        )
    return 1 if any(v.regressed for v in verdicts) else 0


if __name__ == "__main__":
    sys.exit(main())
