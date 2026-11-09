"""A small hybrid retriever whose failure modes can be switched on from its config."""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were will with".split()
)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Config:
    lowercase: bool = True
    remove_stopwords: bool = True
    index_title: bool = True
    doc_max_words: Optional[int] = None
    query_max_words: Optional[int] = None
    k1: float = 1.2
    b: float = 0.75
    query_prefix: bool = True
    normalize: bool = True
    pooling: str = "cls"
    fusion: str = "rrf"
    rrf_k: int = 60
    depth: int = 100
    top_k: int = 10
    drop_docs: float = 0.0
    drop_seed: int = 0


def tokenize(text, config):
    if config.lowercase:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
    else:
        tokens = re.findall(r"[A-Za-z0-9]+", text)
    if config.remove_stopwords:
        tokens = [token for token in tokens if token.lower() not in STOPWORDS]
    return tokens


def document_text(doc, config):
    text = f"{doc['title']} {doc['text']}" if config.index_title else doc["text"]
    if config.doc_max_words is not None:
        text = " ".join(text.split()[: config.doc_max_words])
    return text


def query_text(query, config):
    if config.query_max_words is not None:
        return " ".join(query.split()[: config.query_max_words])
    return query


class BM25:
    """Okapi BM25 over tokenized documents."""

    def __init__(self, documents, k1, b):
        self.k1, self.b = k1, b
        self.size = len(documents)
        self.lengths = np.array([len(tokens) for tokens in documents], dtype=float)
        self.average_length = float(self.lengths.mean())
        postings = defaultdict(list)
        for index, tokens in enumerate(documents):
            for term, count in Counter(tokens).items():
                postings[term].append((index, count))
        self.postings = {
            term: (np.array([i for i, _ in entries]), np.array([c for _, c in entries], dtype=float))
            for term, entries in postings.items()
        }

    def scores(self, tokens):
        scores = np.zeros(self.size)
        for term in tokens:
            if term not in self.postings:
                continue
            indices, counts = self.postings[term]
            idf = np.log(1 + (self.size - indices.size + 0.5) / (indices.size + 0.5))
            norm = self.k1 * (1 - self.b + self.b * self.lengths[indices] / self.average_length)
            scores[indices] += idf * counts * (self.k1 + 1) / (counts + norm)
        return scores


def top_indices(scores, depth):
    """Indices of the best scores; ties keep document order so runs are reproducible."""
    return np.argsort(-scores, kind="stable")[:depth]


def reciprocal_rank_fusion(rankings, k):
    """Fuse ranked index lists; ties keep the order of first appearance."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[int(index)] += 1 / (k + rank)
    return sorted(fused, key=lambda index: -fused[index])


def _l2(matrix):
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


class Pipeline:
    """Retrieve over `doc_ids` given tokenized documents and raw document vectors."""

    def __init__(self, doc_ids, doc_tokens, doc_vectors, config):
        rng = np.random.default_rng(config.drop_seed)
        kept = np.sort(rng.permutation(len(doc_ids))[int(len(doc_ids) * config.drop_docs) :])
        self.config = config
        self.doc_ids = [doc_ids[i] for i in kept]
        self.bm25 = BM25([doc_tokens[i] for i in kept], config.k1, config.b)
        vectors = doc_vectors[kept]
        self.doc_vectors = _l2(vectors) if config.normalize else vectors

    def search(self, query_tokens, query_vector):
        config = self.config
        lists = []
        if config.fusion in ("rrf", "bm25"):
            scores = self.bm25.scores(query_tokens)
            lists.append([index for index in top_indices(scores, config.depth) if scores[index] > 0])
        if config.fusion in ("rrf", "dense"):
            lists.append(top_indices(self.doc_vectors @ query_vector, config.depth))
        order = reciprocal_rank_fusion(lists, config.rrf_k)
        return [self.doc_ids[index] for index in order[: config.top_k]]
