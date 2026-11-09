import math
import random

import pytest
import pytrec_eval

from ragate.metrics import METRICS, ndcg_at, recall_at, reciprocal_rank_at, score_run

QREL = {"a": 2, "b": 1, "c": 0, "d": 1}


def test_recall_ignores_zero_grade_documents():
    assert recall_at(["c", "a", "x"], QREL, 3) == pytest.approx(1 / 3)


def test_recall_respects_cutoff():
    assert recall_at(["x", "a", "b"], QREL, 2) == pytest.approx(1 / 3)
    assert recall_at(["x", "a", "b"], QREL, 3) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_the_first_relevant_document():
    assert reciprocal_rank_at(["c", "x", "b", "a"], QREL, 10) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_past_the_cutoff():
    assert reciprocal_rank_at(["x", "y", "a"], QREL, 2) == 0.0
    assert reciprocal_rank_at(["x", "a"], QREL, 2) == pytest.approx(1 / 2)


def test_ndcg_matches_a_hand_computed_value():
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at(["b", "a", "x"], QREL, 3) == pytest.approx(dcg / ideal)


def test_ndcg_is_one_for_the_ideal_ranking():
    assert ndcg_at(["a", "b", "d", "c"], QREL, 4) == pytest.approx(1.0)


def test_ndcg_truncates_the_ideal_list_to_the_cutoff():
    assert ndcg_at(["a"], QREL, 1) == pytest.approx(1.0)


def test_ndcg_treats_negative_grades_as_zero():
    assert ndcg_at(["n", "a"], {"n": -2, "a": 1}, 2) == pytest.approx(1 / math.log2(3))


def test_score_run_reports_every_metric_per_query():
    scores = score_run({"q": ["a"]}, {"q": {"a": 1}})
    assert set(scores) == set(METRICS)
    assert all(scores[name]["q"] == pytest.approx(1.0) for name in METRICS)


def test_score_run_uses_a_cutoff_of_ten():
    ranking = [f"x{i}" for i in range(10)] + ["a"]
    scores = score_run({"q": ranking}, {"q": {"a": 1}})
    assert all(scores[name]["q"] == 0.0 for name in METRICS)
    hit = score_run({"q": ranking[1:]}, {"q": {"a": 1}})
    assert hit["mrr@10"]["q"] == pytest.approx(1 / 10)
    assert hit["recall@10"]["q"] == 1.0
    assert hit["ndcg@10"]["q"] == pytest.approx(1 / math.log2(11))


def test_score_run_skips_queries_without_relevant_documents():
    scores = score_run({"q": ["a"], "empty": ["a"]}, {"q": {"a": 1}, "empty": {"a": 0}})
    assert set(scores["ndcg@10"]) == {"q"}


def test_score_run_rejects_a_missing_query():
    with pytest.raises(ValueError, match="missing 1"):
        score_run({}, {"q": {"a": 1}})


def test_score_run_rejects_duplicate_documents():
    with pytest.raises(ValueError, match="duplicate"):
        score_run({"q": ["a", "a"]}, {"q": {"a": 1}})


def test_score_run_rejects_judgments_without_any_relevant_document():
    with pytest.raises(ValueError, match="no query"):
        score_run({"q": ["a"]}, {"q": {"a": 0}})


def _random_case(rng):
    documents = [f"d{i}" for i in range(rng.randint(12, 40))]
    judged = rng.sample(documents, rng.randint(3, len(documents)))
    qrel = {doc: rng.choice((0, 0, 1, 2, 3)) for doc in judged}
    qrel[judged[0]] = rng.randint(1, 3)
    return rng.sample(documents, rng.randint(1, len(documents))), qrel


def test_metrics_agree_with_trec_eval_on_random_runs():
    rng = random.Random(20261109)
    qrels, run, rankings = {}, {}, {}
    for number in range(500):
        query = f"q{number}"
        rankings[query], qrels[query] = _random_case(rng)
        top = rankings[query][:10]
        run[query] = {doc: float(len(top) - i) for i, doc in enumerate(top)}
    official = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10", "recip_rank"}).evaluate(run)
    ours = score_run(rankings, qrels)
    for query, expected in official.items():
        assert ours["ndcg@10"][query] == pytest.approx(expected["ndcg_cut_10"], abs=1e-4)
        assert ours["recall@10"][query] == pytest.approx(expected["recall_10"], abs=1e-4)
        assert ours["mrr@10"][query] == pytest.approx(expected["recip_rank"], abs=1e-4)
