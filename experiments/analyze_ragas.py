#!/usr/bin/env python3
"""Compare the Ragas judge with the NLI and lexical detectors on exactly the same responses."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ragate.faithfulness import hallucination_score
from ragate.judge import hallucination_from_faithfulness
from ragate.stats import best_f1_threshold, binary_report, bootstrap_statistic_ci, roc_auc

RESULTS = Path(__file__).resolve().parent.parent / "results"
TASKS = ("QA", "Summary", "Data2txt")
DETECTORS = ("ragas", "nli", "lexical")


def load(name):
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def joined():
    """Ragas rows, each carrying the hallucination score of all three detectors."""
    baseline = {row["id"]: row for row in load("faithfulness_scores.jsonl")}
    latest = {row["id"]: row for row in load("ragas_scores.jsonl")}  # a retried response keeps its last record
    rows = []
    for row in latest.values():
        other = baseline[row["id"]]
        scores = {
            "ragas": hallucination_from_faithfulness(row["faithfulness"]),
            "nli": hallucination_score(other["nli"]),
            "lexical": hallucination_score(other["lexical"]),
        }
        rows.append({**row, "scores": scores})
    return rows


def arrays(rows, detector):
    labels = np.array([row["label_any"] for row in rows], dtype=bool)
    scores = np.array([row["scores"][detector] for row in rows], dtype=float)
    return labels, scores


def block(rows, detector, threshold):
    labels, scores = arrays(rows, detector)
    report = binary_report(labels, scores, threshold)
    report["auc"] = roc_auc(labels, scores)
    report["auc_ci"] = bootstrap_statistic_ci(labels, scores, roc_auc)
    report["n"] = int(labels.size)
    report["positive_rate"] = float(labels.mean())
    return report


def auc_difference(rows, first, second):
    """Paired bootstrap interval of AUC(first) - AUC(second) over the same responses."""
    labels = arrays(rows, first)[0]
    stacked = np.column_stack([arrays(rows, first)[1], arrays(rows, second)[1]])
    point = roc_auc(labels, stacked[:, 0]) - roc_auc(labels, stacked[:, 1])
    low, high = bootstrap_statistic_ci(labels, stacked, lambda l, s: roc_auc(l, s[:, 0]) - roc_auc(l, s[:, 1]))
    return {"delta": point, "ci": [low, high]}


def usage_summary(rows, prices):
    totals = Counter()
    for row in rows:
        totals.update(row["usage"])
    n = max(len(rows), 1)
    summary = {key: totals[key] / n for key in ("calls", "prompt", "cached", "completion")}
    summary["seconds"] = float(np.mean([row["seconds"] for row in rows])) if rows else 0.0
    if prices:
        uncached = summary["prompt"] - summary["cached"]
        per_response = (uncached * prices[0] + summary["cached"] * prices[1] + summary["completion"] * prices[2]) / 1e6
        summary["cost_per_1000_responses"] = per_response * 1000
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=float, nargs=3, metavar=("INPUT", "CACHED", "OUTPUT"),
                        help="price per million tokens for uncached input, cached input and output")
    args = parser.parse_args()
    rows = joined()
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    report = {
        "n_train": len(train),
        "n_test": len(test),
        "judge_failures": dict(Counter(row["error"] for row in rows if row["faithfulness"] is None)),
        "usage_per_response": usage_summary(rows, args.prices),
    }
    for detector in DETECTORS:
        threshold = best_f1_threshold(*arrays(train, detector))
        report[detector] = {"threshold": threshold, "overall": block(test, detector, threshold), "by_task": {}}
        for task in TASKS:
            subset = [row for row in test if row["task"] == task]
            report[detector]["by_task"][task] = block(subset, detector, threshold)
    report["auc_difference"] = {
        "ragas_minus_lexical": auc_difference(test, "ragas", "lexical"),
        "ragas_minus_nli": auc_difference(test, "ragas", "nli"),
    }
    (RESULTS / "ragas.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for detector in DETECTORS:
        overall = report[detector]["overall"]
        print(detector, {key: round(overall[key], 4) for key in ("auc", "f1", "precision", "recall")}, overall["auc_ci"])
    print("auc difference", report["auc_difference"])
    print("failures", report["judge_failures"], "usage", report["usage_per_response"])


if __name__ == "__main__":
    main()
