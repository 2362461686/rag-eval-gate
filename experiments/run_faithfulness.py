#!/usr/bin/env python3
"""Score RAGTruth responses with the NLI and lexical detectors; safe to resume."""
import argparse
import json
import random
from pathlib import Path

from common import load_ragtruth
from ragate.faithfulness import is_claim, lexical_supports, sentence_supports, span_overlaps, split_sentences
from ragate.onnx_models import EntailmentModel

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRAIN_SAMPLE = 3000


def select(rows):
    """All test responses, plus a fixed random sample of train responses for calibration."""
    train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    random.Random(20261109).shuffle(train)
    return train[:TRAIN_SAMPLE] + [row for row in rows if row["split"] == "test"]


def score_row(row, entail):
    claims = [(s, e, text) for s, e, text in split_sentences(row["response"]) if is_claim(text)]
    sentences = [text for _, _, text in claims]
    strict = [label for label in row["labels"] if not (label["implicit_true"] or label["due_to_null"])]
    return {
        "id": row["id"],
        "split": row["split"],
        "task": row["task"],
        "model": row["model"],
        "label_any": bool(row["labels"]),
        "label_strict": bool(strict),
        "nli": sentence_supports(sentences, row["evidence"], entail),
        "lexical": lexical_supports(sentences, row["evidence"]),
        "sentence_labels": [span_overlaps(s, e, row["labels"]) for s, e, _ in claims],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory holding response.jsonl and source_info.jsonl")
    parser.add_argument("--model", type=Path, required=True, help="directory holding the NLI model")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new responses")
    args = parser.parse_args()
    output = RESULTS / "faithfulness_scores.jsonl"
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            done = {json.loads(line)["id"] for line in handle}
    pending = [row for row in select(load_ragtruth(args.data)) if row["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    entail = EntailmentModel(args.model, threads=args.threads)
    RESULTS.mkdir(exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for count, row in enumerate(pending, start=1):
            handle.write(json.dumps(score_row(row, entail)) + "\n")
            if count % 200 == 0:
                handle.flush()
                print(f"{count}/{len(pending)}", flush=True)
    print(f"done, {len(done) + len(pending)} scored", flush=True)


if __name__ == "__main__":
    main()
