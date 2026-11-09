#!/usr/bin/env python3
"""Calibrate detector thresholds on train, then report once on the untouched test split."""
import json
from pathlib import Path

import numpy as np

from ragate.faithfulness import hallucination_score
from ragate.stats import best_f1_threshold, binary_report, bootstrap_statistic_ci, roc_auc

RESULTS = Path(__file__).resolve().parent.parent / "results"
TASKS = ("QA", "Summary", "Data2txt")


def load():
    with (RESULTS / "faithfulness_scores.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def arrays(rows, detector, label):
    labels = np.array([row[label] for row in rows], dtype=bool)
    scores = np.array([hallucination_score(row[detector]) for row in rows], dtype=float)
    return labels, scores


def f1_at(threshold):
    return lambda labels, scores: binary_report(labels, scores, threshold)["f1"]


def block(labels, scores, threshold):
    report = binary_report(labels, scores, threshold)
    report["auc"] = roc_auc(labels, scores)
    report["auc_ci"] = bootstrap_statistic_ci(labels, scores, roc_auc)
    report["f1_ci"] = bootstrap_statistic_ci(labels, scores, f1_at(threshold))
    report["n"] = int(labels.size)
    report["positive_rate"] = float(labels.mean())
    return report


def detector_report(train, test, detector, label):
    threshold = best_f1_threshold(*arrays(train, detector, label))
    labels, scores = arrays(test, detector, label)
    report = {"threshold": threshold, "overall": block(labels, scores, threshold)}
    report["by_task"] = {}
    for task in TASKS:
        subset = [row for row in test if row["task"] == task]
        report["by_task"][task] = block(*arrays(subset, detector, label), threshold)
    report["by_model"] = {}
    for model in sorted({row["model"] for row in test}):
        subset = [row for row in test if row["model"] == model]
        report["by_model"][model] = block(*arrays(subset, detector, label), threshold)
    return report


def transfer_matrix(train, test, detector, label):
    """F1 on each test task, for a threshold calibrated on each train task."""
    matrix = {}
    for source in TASKS:
        threshold = best_f1_threshold(*arrays([r for r in train if r["task"] == source], detector, label))
        matrix[source] = {"threshold": threshold}
        for target in TASKS:
            subset = [r for r in test if r["task"] == target]
            matrix[source][target] = binary_report(*arrays(subset, detector, label), threshold)["f1"]
    return matrix


def sentence_level(train, test, detector):
    def flat(rows):
        labels = np.array([x for row in rows for x in row["sentence_labels"]], dtype=bool)
        scores = 1.0 - np.array([x for row in rows for x in row[detector]], dtype=float)
        return labels, scores

    threshold = best_f1_threshold(*flat(train))
    return {"threshold": threshold, **block(*flat(test), threshold)}


def main():
    rows = load()
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    labels = np.array([row["label_any"] for row in test], dtype=bool)
    report = {
        "dataset": {"name": "RAGTruth", "train_for_calibration": len(train), "test": len(test)},
        "trivial_all_positive": binary_report(labels, np.ones(labels.size), 0.5),
    }
    for detector in ("nli", "lexical"):
        report[detector] = detector_report(train, test, detector, "label_any")
        report[detector]["transfer"] = transfer_matrix(train, test, detector, "label_any")
        report[detector]["sentence_level"] = sentence_level(train, test, detector)
        report[detector]["strict_labels"] = detector_report(train, test, detector, "label_strict")["overall"]
    (RESULTS / "faithfulness.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for detector in ("nli", "lexical"):
        overall = report[detector]["overall"]
        print(detector, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in overall.items() if k not in ("auc_ci", "f1_ci")})
    print("all positive", report["trivial_all_positive"])


if __name__ == "__main__":
    main()
