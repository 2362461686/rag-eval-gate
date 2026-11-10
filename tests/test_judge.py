import json

import pytest

from ragate.judge import add_usage, hallucination_from_faithfulness, judge_inputs, read_settings, usage_from_body


def test_qa_uses_the_question_and_the_passages():
    source = {"question": "butcher shop phone number", "passages": "passage 1: (510) 889-8690"}
    assert judge_inputs("QA", source, "ignored") == ("butcher shop phone number", ["passage 1: (510) 889-8690"])


def test_summary_keeps_only_the_instruction_line():
    prompt = "Summarize the following news within 116 words:\nSeventy years ago, Anne Frank died."
    request, contexts = judge_inputs("Summary", "Seventy years ago, Anne Frank died.", prompt)
    assert request == "Summarize the following news within 116 words:"
    assert contexts == ["Seventy years ago, Anne Frank died."]


def test_data2txt_separates_the_instruction_from_the_structured_data():
    source = {"name": "Apna Indian Kitchen", "city": "Santa Barbara"}
    prompt = (
        "Instruction:\nWrite an objective overview about the following local business. "
        "Don't make up information. Structured data:\n{'name': 'Apna Indian Kitchen'}\nOverview:"
    )
    request, contexts = judge_inputs("Data2txt", source, prompt)
    assert request == "Write an objective overview about the following local business. Don't make up information."
    assert json.loads(contexts[0]) == source


@pytest.mark.parametrize(
    ("task", "prompt"),
    [
        ("Summary", "Summarise this:\ntext"),
        ("Data2txt", "Write an overview.\n{'name': 'x'}"),
        ("Translation", "anything"),
    ],
)
def test_unrecognised_prompts_fail_loudly(task, prompt):
    with pytest.raises(ValueError):
        judge_inputs(task, {"name": "x"} if task == "Data2txt" else "text", prompt)


def test_read_settings_skips_comments_and_keeps_equals_signs_in_values(tmp_path):
    path = tmp_path / "judge.env"
    path.write_text(
        "# comment\n\nJUDGE_BASE_URL = https://api.example.com\nJUDGE_MODEL=judge-1\nJUDGE_API_KEY=sk-a=b\n",
        encoding="utf-8",
    )
    assert read_settings(path) == {
        "JUDGE_BASE_URL": "https://api.example.com",
        "JUDGE_MODEL": "judge-1",
        "JUDGE_API_KEY": "sk-a=b",
    }


def test_read_settings_names_the_missing_key_without_echoing_values(tmp_path):
    path = tmp_path / "judge.env"
    path.write_text("JUDGE_BASE_URL=https://secret-host.example\nJUDGE_MODEL=judge-1\nJUDGE_API_KEY=\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        read_settings(path)
    assert "JUDGE_API_KEY" in str(error.value)
    assert "secret-host" not in str(error.value)


def test_usage_reads_deepseek_cache_hits():
    body = {"usage": {"prompt_tokens": 1200, "completion_tokens": 300, "prompt_cache_hit_tokens": 1024}}
    assert usage_from_body(body) == {"calls": 1, "prompt": 1200, "cached": 1024, "completion": 300}


def test_usage_reads_openai_cached_tokens_and_tolerates_missing_usage():
    body = {"usage": {"prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 32}}}
    assert usage_from_body(body) == {"calls": 1, "prompt": 50, "cached": 32, "completion": 7}
    assert usage_from_body({}) == {"calls": 1, "prompt": 0, "cached": 0, "completion": 0}


def test_add_usage_sums_every_counter():
    total = add_usage({}, {"calls": 1, "prompt": 10, "cached": 0, "completion": 3})
    add_usage(total, {"calls": 1, "prompt": 5, "cached": 4, "completion": 2})
    assert total == {"calls": 2, "prompt": 15, "cached": 4, "completion": 5}


@pytest.mark.parametrize(("faithfulness", "expected"), [(1.0, 0.0), (0.75, 0.25), (0.0, 1.0), (None, 1.0)])
def test_hallucination_is_the_unfaithful_share_and_fails_closed(faithfulness, expected):
    assert hallucination_from_faithfulness(faithfulness) == pytest.approx(expected)
