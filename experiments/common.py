"""Loaders for the public datasets; the data itself stays outside the repository."""
import csv
import json
import random
from pathlib import Path

TRAIN_SAMPLE = 3000


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_beir(root, split):
    """Return (corpus, queries, qrels) for one BEIR split, keeping only judged queries."""
    root = Path(root)
    corpus = {row["_id"]: {"title": row.get("title", ""), "text": row["text"]} for row in _jsonl(root / "corpus.jsonl")}
    all_queries = {row["_id"]: row["text"] for row in _jsonl(root / "queries.jsonl")}
    qrels = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return corpus, {query: all_queries[query] for query in qrels}, qrels


def flatten_evidence(source):
    """RAGTruth stores evidence as text, or as nested JSON for QA and data-to-text."""
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        return "\n".join(f"{key}: {flatten_evidence(value)}" for key, value in source.items())
    if isinstance(source, list):
        return "\n".join(flatten_evidence(value) for value in source)
    return str(source)


def load_ragtruth(root):
    """Return good-quality responses joined with their task type and evidence."""
    root = Path(root)
    sources = {row["source_id"]: row for row in _jsonl(root / "source_info.jsonl")}
    rows = []
    for row in _jsonl(root / "response.jsonl"):
        if row["quality"] != "good":
            continue
        source = sources[row["source_id"]]
        rows.append({**row, "task": source["task_type"], "evidence": flatten_evidence(source["source_info"]), "source": source})
    return rows


def calibration_and_test(rows):
    """A fixed random sample of train responses for calibration, in that random order, plus every test response."""
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    random.Random(20261109).shuffle(train)
    return train[:TRAIN_SAMPLE], [row for row in rows if row["split"] == "test"]
