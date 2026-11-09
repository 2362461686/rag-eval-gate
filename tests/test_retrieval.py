import math

import numpy as np
import pytest

from ragate.retrieval import (
    BM25,
    Config,
    Pipeline,
    document_text,
    query_text,
    reciprocal_rank_fusion,
    tokenize,
    top_indices,
)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cell-Cycle of T2 cells", Config()) == ["cell", "cycle", "t2", "cells"]


def test_tokenize_without_lowercasing_keeps_case_and_still_drops_stopwords():
    assert tokenize("The Cell of T2", Config(lowercase=False)) == ["Cell", "T2"]


def test_tokenize_can_keep_stopwords():
    assert tokenize("the cell", Config(remove_stopwords=False)) == ["the", "cell"]


def test_document_text_includes_the_title_by_default():
    assert document_text({"title": "T", "text": "body"}, Config()) == "T body"


def test_document_text_can_drop_the_title():
    assert document_text({"title": "T", "text": "body"}, Config(index_title=False)) == "body"


def test_document_text_truncates_to_the_word_limit():
    assert document_text({"title": "a", "text": "b c d"}, Config(doc_max_words=2)) == "a b"


def test_query_text_is_unchanged_without_a_limit():
    assert query_text("one two  three", Config()) == "one two  three"


def test_query_text_truncates_to_the_word_limit():
    assert query_text("one two three", Config(query_max_words=2)) == "one two"


def _bm25_term(tf, df, size, length, average, k1=1.2, b=0.75):
    idf = math.log(1 + (size - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))


def test_bm25_matches_the_okapi_formula():
    documents = [["apple", "apple", "pear"], ["pear"], ["plum", "plum", "plum", "plum"]]
    scores = BM25(documents, 1.2, 0.75).scores(["apple", "pear"])
    average = 8 / 3
    assert scores[0] == pytest.approx(_bm25_term(2, 1, 3, 3, average) + _bm25_term(1, 2, 3, 3, average))
    assert scores[1] == pytest.approx(_bm25_term(1, 2, 3, 1, average))
    assert scores[2] == 0.0


def test_bm25_uses_the_true_average_length_of_short_documents():
    scores = BM25([["a"], ["b"]], 1.2, 0.75).scores(["a"])
    assert scores[0] == pytest.approx(_bm25_term(1, 1, 2, 1, 1.0))


def test_bm25_uses_the_given_parameters():
    documents = [["a", "a", "b"], ["b"], ["c", "c", "c"]]
    scores = BM25(documents, 2.0, 0.5).scores(["a"])
    assert scores[0] == pytest.approx(_bm25_term(2, 1, 3, 3, 7 / 3, k1=2.0, b=0.5))


def test_bm25_counts_a_repeated_query_term_twice():
    index = BM25([["a"], ["b"]], 1.2, 0.75)
    assert index.scores(["a", "a"])[0] == pytest.approx(2 * index.scores(["a"])[0])


def test_bm25_ignores_unknown_terms_and_handles_empty_documents():
    assert BM25([[], []], 1.2, 0.75).scores(["a"]).tolist() == [0.0, 0.0]


def test_top_indices_sorts_descending_and_keeps_document_order_on_ties():
    assert top_indices(np.array([0.1, 0.9, 0.5, 0.9]), 3).tolist() == [1, 3, 2]


def test_reciprocal_rank_fusion_matches_a_hand_computed_order():
    # doc 5: 1/61 + 1/62; doc 7: 1/62 + 1/61; doc 9: 1/63 (ties keep first appearance)
    assert reciprocal_rank_fusion([[5, 7, 9], [7, 5]], 60) == [5, 7, 9]


def test_reciprocal_rank_fusion_rewards_agreement_over_a_single_top_rank():
    assert reciprocal_rank_fusion([[1, 2], [3, 2], [4, 2]], 60)[0] == 2


def test_reciprocal_rank_fusion_depends_on_k():
    # one first place against two third places: a small k favours the first place
    lists = [[1, 8, 9], [5, 6, 2], [3, 4, 2]]
    assert reciprocal_rank_fusion(lists, 0)[0] == 1
    assert reciprocal_rank_fusion(lists, 1000)[0] == 2


IDS = ["d0", "d1", "d2", "d3"]
TOKENS = [["alpha"], ["beta"], ["gamma"], ["alpha", "beta"]]
VECTORS = np.array([[0.1, 0.0], [0.0, 3.0], [2.0, 2.0], [0.0, 0.5]])


def _search(config, tokens=("alpha",), vector=(0.0, 1.0)):
    return Pipeline(IDS, TOKENS, VECTORS, config).search(list(tokens), np.array(vector))


def test_bm25_only_fusion_ignores_the_dense_list_and_drops_unmatched_documents():
    assert _search(Config(fusion="bm25")) == ["d0", "d3"]


def test_dense_only_fusion_ignores_the_bm25_list():
    assert _search(Config(fusion="dense")) == ["d1", "d3", "d2", "d0"]


def test_hybrid_fusion_uses_both_lists():
    assert _search(Config())[0] == "d3"


def test_unnormalized_vectors_change_the_dense_order():
    assert _search(Config(fusion="dense", normalize=False)) == ["d1", "d2", "d3", "d0"]


def test_unnormalized_query_vector_is_left_alone():
    pipeline = Pipeline(IDS, TOKENS, VECTORS, Config(fusion="dense", normalize=False, top_k=4))
    tilted = pipeline.search([], np.array([1.0, 0.2]))
    assert tilted == ["d2", "d1", "d0", "d3"]


def test_top_k_limits_the_results():
    assert len(_search(Config(top_k=2))) == 2


def test_depth_limits_each_candidate_list():
    assert _search(Config(fusion="bm25", depth=1)) == ["d0"]


def test_rrf_k_is_forwarded():
    ids = ["d0", "d1", "d2"]
    tokens = [["x", "x", "x"], ["y"], ["x"]]
    vectors = np.array([[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]])
    query = np.array([1.0, 0.0])
    assert Pipeline(ids, tokens, vectors, Config(rrf_k=0, depth=2)).search(["x"], query)[0] == "d0"
    assert Pipeline(ids, tokens, vectors, Config(rrf_k=1000, depth=2)).search(["x"], query)[0] == "d2"


def test_dropping_documents_removes_that_fraction_reproducibly():
    first = Pipeline(IDS, TOKENS, VECTORS, Config(drop_docs=0.5, drop_seed=3))
    second = Pipeline(IDS, TOKENS, VECTORS, Config(drop_docs=0.5, drop_seed=3))
    assert len(first.doc_ids) == 2 and first.doc_ids == second.doc_ids
    assert first.doc_ids == sorted(first.doc_ids)


def test_dropping_documents_depends_on_the_seed():
    many = [f"d{i}" for i in range(50)]
    tokens = [["x"]] * 50
    vectors = np.ones((50, 2))
    first = Pipeline(many, tokens, vectors, Config(drop_docs=0.5, drop_seed=1)).doc_ids
    second = Pipeline(many, tokens, vectors, Config(drop_docs=0.5, drop_seed=2)).doc_ids
    assert first != second


def test_kept_documents_keep_their_own_tokens_and_vectors():
    pipeline = Pipeline(IDS, TOKENS, VECTORS, Config(drop_docs=0.5, drop_seed=3, fusion="bm25", top_k=4))
    for doc_id, index in zip(pipeline.doc_ids, range(2)):
        original = IDS.index(doc_id)
        assert pipeline.bm25.lengths[index] == len(TOKENS[original])
    assert pipeline.doc_vectors.shape == (2, 2)
