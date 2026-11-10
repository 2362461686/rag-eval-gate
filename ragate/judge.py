"""Glue for scoring RAGTruth with an LLM judge; kept free of LLM client imports so it stays testable."""
import json
import re
from pathlib import Path

SUMMARY_INSTRUCTION = re.compile(r"(Summarize the following news within \d+ words:)\n")
DATA2TXT_INSTRUCTION = re.compile(r"Instruction:\n(.+?) Structured data:\n", re.S)
REQUIRED_SETTINGS = ("JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY")


def judge_inputs(task, source_info, prompt):
    """Split a RAGTruth source into the user request and the evidence the response had to stay within."""
    if task == "QA":
        return source_info["question"], [source_info["passages"]]
    if task == "Summary":
        match = SUMMARY_INSTRUCTION.match(prompt)
        if match is None:
            raise ValueError("unrecognised Summary prompt")
        return match.group(1), [source_info]
    if task == "Data2txt":
        match = DATA2TXT_INSTRUCTION.match(prompt)
        if match is None:
            raise ValueError("unrecognised Data2txt prompt")
        return match.group(1), [json.dumps(source_info, ensure_ascii=False)]
    raise ValueError(f"unknown task {task!r}")


def read_settings(path):
    """KEY=VALUE lines from the judge settings file; error messages never echo a value."""
    settings = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    missing = [key for key in REQUIRED_SETTINGS if not settings.get(key)]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    return settings


def usage_from_body(body):
    """Token counts from one chat-completion response; providers report cached input differently."""
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "calls": 1,
        "prompt": int(usage.get("prompt_tokens") or 0),
        "cached": int(usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0),
        "completion": int(usage.get("completion_tokens") or 0),
    }


def add_usage(total, usage):
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value
    return total


def hallucination_from_faithfulness(faithfulness):
    """1 - faithfulness; a judge that returned no score counts as fully suspect (fail closed)."""
    return 1.0 if faithfulness is None else 1.0 - faithfulness
