"""Per-query retrieval metrics, defined the way trec_eval defines them."""
import math

CUTOFF = 10
METRICS = ("ndcg@10", "recall@10", "mrr@10")


def _relevant(qrel):
    return {doc for doc, grade in qrel.items() if grade > 0}


def _dcg(gains):
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def recall_at(ranking, qrel, k):
    relevant = _relevant(qrel)
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def reciprocal_rank_at(ranking, qrel, k):
    relevant = _relevant(qrel)
    for rank, doc in enumerate(ranking[:k], start=1):
        if doc in relevant:
            return 1 / rank
    return 0.0


def ndcg_at(ranking, qrel, k):
    gains = [max(qrel.get(doc, 0), 0) for doc in ranking[:k]]
    ideal = sorted((max(grade, 0) for grade in qrel.values()), reverse=True)[:k]
    return _dcg(gains) / _dcg(ideal)


def score_run(run, qrels):
    """Return {metric: {query: value}} for every query with a relevant document."""
    judged = {query: qrel for query, qrel in qrels.items() if _relevant(qrel)}
    if not judged:
        raise ValueError("no query has a relevant document")
    missing = judged.keys() - run.keys()
    if missing:
        raise ValueError(f"run is missing {len(missing)} judged queries")
    scores = {name: {} for name in METRICS}
    for query, qrel in judged.items():
        ranking = run[query]
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"duplicate documents in the ranking for query {query}")
        scores["ndcg@10"][query] = ndcg_at(ranking, qrel, CUTOFF)
        scores["recall@10"][query] = recall_at(ranking, qrel, CUTOFF)
        scores["mrr@10"][query] = reciprocal_rank_at(ranking, qrel, CUTOFF)
    return scores
