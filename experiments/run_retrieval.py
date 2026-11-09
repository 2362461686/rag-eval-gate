#!/usr/bin/env python3
"""Run the hybrid retriever on BEIR SciFact and measure what the gate can and cannot see."""
import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from common import load_beir
from ragate.gate import compare
from ragate.metrics import score_run
from ragate.onnx_models import Encoder
from ragate.retrieval import QUERY_PREFIX, Config, Pipeline, document_text, query_text, tokenize

RESULTS = Path(__file__).resolve().parent.parent / "results"
BASE = Config()
FAULTS = {
    "tokenizer_lost_lowercasing": ("BM25 分词丢了小写归一", replace(BASE, lowercase=False)),
    "stopwords_kept": ("BM25 停用词过滤失效", replace(BASE, remove_stopwords=False)),
    "ingest_lost_title": ("入库时丢失文档标题", replace(BASE, index_title=False)),
    "chunk_truncated_32_words": ("切块错误，只保留前 32 词", replace(BASE, doc_max_words=32)),
    "query_truncated_8_words": ("查询被截断到 8 词", replace(BASE, query_max_words=8)),
    "bge_query_prefix_missing": ("漏掉 BGE 查询前缀", replace(BASE, query_prefix=False)),
    "embeddings_not_normalized": ("向量未归一化", replace(BASE, normalize=False)),
    "wrong_pooling_mean": ("BGE 误用 mean pooling", replace(BASE, pooling="mean")),
    "silent_fallback_to_bm25": ("向量服务静默降级，只剩 BM25", replace(BASE, fusion="bm25")),
    "silent_fallback_to_dense": ("关键词索引静默降级，只剩向量", replace(BASE, fusion="dense")),
    "rrf_k_zero": ("RRF 平滑常数被置 0", replace(BASE, rrf_k=0)),
    "candidate_depth_10": ("召回深度从 100 掉到 10", replace(BASE, depth=10)),
    "ingest_lost_10pct_docs": ("入库丢失 10% 文档", replace(BASE, drop_docs=0.10)),
    "top_k_off_by_one": ("TopK 差一，只返回 9 条", replace(BASE, top_k=9)),
    "bm25_tf_saturation_broken": ("BM25 k1=0，词频失效", replace(BASE, k1=0.0)),
    "bm25_length_norm_broken": ("BM25 b=0，长度归一失效", replace(BASE, b=0.0)),
}
BENIGN = {
    "bm25_k1_1.5": replace(BASE, k1=1.5),
    "bm25_b_0.7": replace(BASE, b=0.7),
    "rrf_k_50": replace(BASE, rrf_k=50),
    "candidate_depth_200": replace(BASE, depth=200),
}
DROP_LEVELS = (0.01, 0.02, 0.05, 0.10, 0.20)
DROP_SEEDS = 20
NULL_TRIALS = 300


class Workspace:
    """Caches embeddings on disk and tokens in memory, so each config costs little."""

    def __init__(self, corpus, queries, qrels, encoder, cache):
        self.ids = sorted(corpus)
        self.corpus, self.query_ids, self.queries, self.qrels = corpus, sorted(queries), queries, qrels
        self.encoder, self.cache = encoder, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.tokens = {}

    def embed(self, texts, pooling):
        digest = hashlib.sha256("\0".join([pooling, *texts]).encode()).hexdigest()[:24]
        path = self.cache / f"{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.encoder.encode(texts, pooling)
        np.save(path, vectors)
        return vectors

    def run(self, config):
        texts = [document_text(self.corpus[doc], config) for doc in self.ids]
        key = (config.lowercase, config.remove_stopwords, config.index_title, config.doc_max_words)
        if key not in self.tokens:
            self.tokens[key] = [tokenize(text, config) for text in texts]
        pipeline = Pipeline(self.ids, self.tokens[key], self.embed(texts, config.pooling), config)
        queries = [query_text(self.queries[q], config) for q in self.query_ids]
        prefix = QUERY_PREFIX if config.query_prefix else ""
        vectors = self.embed([prefix + q for q in queries], config.pooling)
        return {
            qid: pipeline.search(tokenize(text, config), vector)
            for qid, text, vector in zip(self.query_ids, queries, vectors)
        }

    def scores(self, config):
        return score_run(self.run(config), self.qrels)


def means(scores):
    return {name: float(np.mean(list(values.values()))) for name, values in scores.items()}


def verdict_rows(verdicts):
    return [asdict(v) for v in verdicts]


def null_false_alarms(baseline, alternative):
    """Randomize the signs of real per-query deltas: every gate failure here is a false alarm."""
    rng = np.random.default_rng(20261109)
    queries = sorted(next(iter(baseline.values())))
    alarms = 0
    for trial in range(NULL_TRIALS):
        flips = dict(zip(queries, rng.choice((-1.0, 1.0), size=len(queries))))
        candidate = {
            name: {q: baseline[name][q] + flips[q] * (alternative[name][q] - baseline[name][q]) for q in queries}
            for name in baseline
        }
        alarms += any(v.regressed for v in compare(baseline, candidate, n_resamples=2000, seed=trial))
    return alarms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory holding the extracted scifact dataset")
    parser.add_argument("--models", type=Path, required=True, help="directory holding bge/ and nli/")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    corpus, queries, qrels = load_beir(args.data / "scifact", "test")
    encoder = Encoder(args.models / "bge", threads=args.threads)
    space = Workspace(corpus, queries, qrels, encoder, args.data / "embedding-cache")

    baseline = space.scores(BASE)
    print("baseline", means(baseline), flush=True)
    report = {
        "dataset": {"name": "BEIR SciFact (test)", "documents": len(corpus), "queries": len(next(iter(baseline.values())))},
        "gate": {"margin": 0.01, "alpha": 0.05, "correction": "holm", "test": "paired sign-flip"},
        "baseline": {"config": asdict(BASE), "means": means(baseline)},
    }

    report["retrievers"] = {}
    for name, fusion in (("bm25", "bm25"), ("dense", "dense"), ("hybrid_rrf", "rrf")):
        report["retrievers"][name] = means(space.scores(replace(BASE, fusion=fusion)))

    report["faults"] = []
    for name, (description, config) in FAULTS.items():
        verdicts = compare(baseline, space.scores(config))
        detected = any(v.regressed for v in verdicts)
        report["faults"].append({"name": name, "description": description, "detected": detected, "verdicts": verdict_rows(verdicts)})
        print(("DETECTED" if detected else "missed  "), name, [round(v.delta, 4) for v in verdicts], flush=True)

    report["benign"] = []
    for name, config in BENIGN.items():
        verdicts = compare(baseline, space.scores(config))
        alarmed = any(v.regressed for v in verdicts)
        report["benign"].append({"name": name, "alarmed": alarmed, "verdicts": verdict_rows(verdicts)})
        print(("ALARM   " if alarmed else "quiet   "), name, [round(v.delta, 4) for v in verdicts], flush=True)

    alternative = space.scores(replace(BASE, fusion="bm25"))
    alarms = null_false_alarms(baseline, alternative)
    report["null_simulation"] = {"trials": NULL_TRIALS, "false_alarms": alarms, "nominal_alpha": 0.05}
    print("null false alarms", alarms, "/", NULL_TRIALS, flush=True)

    report["sensitivity"] = []
    for level in DROP_LEVELS:
        hits, drops = 0, []
        for seed in range(DROP_SEEDS):
            verdicts = compare(baseline, space.scores(replace(BASE, drop_docs=level, drop_seed=seed)), n_resamples=2000)
            hits += any(v.regressed for v in verdicts)
            drops.append(min(v.delta for v in verdicts))
        report["sensitivity"].append({"docs_lost": level, "seeds": DROP_SEEDS, "detected": hits, "mean_worst_delta": float(np.mean(drops))})
        print("docs lost", level, "detected", hits, "/", DROP_SEEDS, flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    runs = RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "baseline.json").write_text(json.dumps({"metrics": baseline}), encoding="utf-8")
    (runs / "candidate_silent_fallback.json").write_text(json.dumps({"metrics": alternative}), encoding="utf-8")


if __name__ == "__main__":
    main()
