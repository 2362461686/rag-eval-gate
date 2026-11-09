import pytest

from ragate.faithfulness import (
    evidence_windows,
    hallucination_score,
    is_claim,
    lexical_supports,
    nearest_windows,
    overlap,
    sentence_supports,
    span_overlaps,
    split_sentences,
)


def test_split_sentences_returns_offsets_into_the_original_text():
    text = "First claim here. Second claim here!\nThird claim? 4 is a number."
    spans = split_sentences(text)
    assert [s for _, _, s in spans] == ["First claim here.", "Second claim here!", "Third claim?", "4 is a number."]
    assert all(text[start:end] == sentence for start, end, sentence in spans)


def test_split_sentences_keeps_offsets_mid_line_after_a_newline():
    text = "Alpha one two.\nBeta one two. Gamma one two. Delta one two."
    spans = split_sentences(text)
    assert len(spans) == 4
    assert all(text[start:end] == sentence for start, end, sentence in spans)


def test_split_sentences_does_not_break_inside_decimals_or_lowercase_continuations():
    assert [s for _, _, s in split_sentences("It rose 3.5 percent. e.g. this stays")] == [
        "It rose 3.5 percent. e.g. this stays"
    ]


def test_split_sentences_breaks_before_quotes_and_brackets():
    assert len(split_sentences('He left. "She stayed." (They ran.) [Done.]')) == 4


def test_split_sentences_drops_blank_lines_and_empty_text():
    assert [s for _, _, s in split_sentences("Alpha beta.\n\n  \nGamma delta.")] == ["Alpha beta.", "Gamma delta."]
    assert split_sentences("") == []


def test_split_sentences_strips_padding():
    assert split_sentences("  padded words  ")[0][2] == "padded words"


@pytest.mark.parametrize(
    "sentence",
    ["Sure, here is the summary of it", "Certainly the answer follows below", "Of course we can do that", "Here is what it says now", "Here are the main points now", "I hope this helps you out", "Thank you for asking me that", "The main points are these four:"],
)
def test_boilerplate_is_not_a_claim(sentence):
    assert not is_claim(sentence)


def test_is_claim_needs_four_words():
    assert not is_claim("Too short claim.")
    assert is_claim("Four words make claim.")


def test_a_sentence_merely_containing_boilerplate_is_a_claim():
    assert is_claim("The vendor said sure about delivery.")


def test_short_evidence_is_a_single_window():
    assert evidence_windows("one  two three") == ["one two three"]


def test_empty_evidence_has_no_windows():
    assert evidence_windows("   ") == []


def test_evidence_of_exactly_the_window_size_is_one_window():
    assert len(evidence_windows(" ".join(map(str, range(4))), size=4, stride=2)) == 1


def test_windows_overlap_and_reach_the_end():
    tokens = [str(i) for i in range(10)]
    assert evidence_windows(" ".join(tokens), size=4, stride=2) == [
        "0 1 2 3",
        "2 3 4 5",
        "4 5 6 7",
        "6 7 8 9",
    ]


def test_windows_cover_a_tail_that_the_stride_would_miss():
    windows = evidence_windows(" ".join(str(i) for i in range(11)), size=4, stride=3)
    assert windows[-1] == "7 8 9 10"
    assert windows[0] == "0 1 2 3"


def test_window_defaults_are_sixty_words_with_a_thirty_word_stride():
    windows = evidence_windows(" ".join(str(i) for i in range(100)))
    assert len(windows[0].split()) == 60
    assert windows[1].split()[0] == "30"


def test_overlap_is_the_share_of_distinct_sentence_words():
    assert overlap("Alpha beta beta gamma", "alpha, GAMMA and more") == pytest.approx(2 / 3)


def test_overlap_of_an_empty_sentence_is_zero():
    assert overlap("...", "alpha") == 0.0


def test_nearest_windows_ranks_by_overlap():
    windows = ["x y", "alpha x", "alpha beta"]
    assert nearest_windows("alpha beta", windows, top=2) == ["alpha beta", "alpha x"]


def test_nearest_windows_defaults_to_three():
    assert len(nearest_windows("a", ["a", "b", "c", "d"])) == 3


def _entail_by_lookup(table):
    def entail(pairs):
        return [table[pair] for pair in pairs]

    return entail


def test_sentence_support_takes_the_best_window_for_each_sentence():
    calls = []

    def entail(pairs):
        calls.extend(pairs)
        return [0.2 if pair[1] == "beta claim" else (0.9 if pair[0].startswith("alpha") else 0.1) for pair in pairs]

    evidence = "alpha claim " + "pad " * 58 + "other words here"
    supports = sentence_supports(["alpha claim", "beta claim"], evidence, entail, top=2)
    assert supports == pytest.approx([0.9, 0.2])
    assert all(hypothesis in ("alpha claim", "beta claim") for _, hypothesis in calls)
    assert len(calls) == 4


def test_sentence_support_passes_the_window_as_the_premise():
    seen = []
    sentence_supports(["a claim"], "the evidence", lambda pairs: seen.extend(pairs) or [1.0] * len(pairs))
    assert seen == [("the evidence", "a claim")]


def test_sentence_support_without_sentences_never_calls_the_model():
    def explode(pairs):
        raise AssertionError("model should not run")

    assert sentence_supports([], "evidence", explode) == []


def test_sentence_support_without_evidence_is_zero():
    def explode(pairs):
        raise AssertionError("model should not run")

    assert sentence_supports(["a claim"], "", explode) == [0.0]


def test_lexical_support_reads_the_whole_evidence():
    assert lexical_supports(["alpha zeta"], "alpha " + "pad " * 500 + "zeta") == [1.0]


def test_hallucination_score_follows_the_weakest_claim():
    assert hallucination_score([0.9, 0.2, 0.7]) == pytest.approx(0.8)


def test_hallucination_score_without_claims_is_zero():
    assert hallucination_score([]) == 0.0


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [(0, 5, False), (0, 6, True), (5, 15, True), (9, 20, True), (10, 20, False), (6, 7, True)],
)
def test_span_overlap_treats_ends_as_exclusive(start, end, expected):
    assert span_overlaps(start, end, [{"start": 5, "end": 10}]) is expected


def test_span_overlap_with_no_labels_is_false():
    assert span_overlaps(0, 10, []) is False
