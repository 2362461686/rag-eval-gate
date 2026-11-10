#!/usr/bin/env python3
"""Score RAGTruth responses with the Ragas faithfulness judge (an LLM); safe to resume.

Train responses come in the same fixed random order as run_faithfulness.py, so a small --limit
on --split train is a pilot that never touches the test split. Test responses are shuffled with
a fixed seed, so any prefix of them is a random sample.
"""
import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # Ragas otherwise posts usage events to its vendor

import argparse
import asyncio
import contextvars
import json
import math
import random
import time
from pathlib import Path

from openai import APIStatusError, AsyncOpenAI, DefaultAsyncHttpxClient
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from common import calibration_and_test, load_ragtruth
from ragate.judge import add_usage, judge_inputs, read_settings, usage_from_body

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUTPUT = RESULTS / "ragas_scores.jsonl"
SETTINGS = Path.home() / ".config" / "rag-eval-gate" / "judge.env"
# Ragas defaults to 1024 output tokens, which truncates the verdict list of a long response.
MAX_TOKENS = 4096
# This many failures in a row means the provider is down or the balance ran out: stop paying for nothing.
MAX_CONSECUTIVE_FAILURES = 10
CURRENT_USAGE = contextvars.ContextVar("current_usage")


async def record_usage(response):
    """httpx response hook: charge each completed chat call to the response being scored."""
    usage = CURRENT_USAGE.get(None)
    if usage is not None and response.status_code == 200 and response.request.url.path.endswith("/chat/completions"):
        await response.aread()
        add_usage(usage, usage_from_body(response.json()))


def describe(error):
    """Exception type and HTTP status only: provider messages can echo part of the key."""
    if isinstance(error, APIStatusError):
        return f"{type(error).__name__}:{error.status_code}"
    return type(error).__name__


async def score(row, metric, semaphore):
    user_input, contexts = judge_inputs(row["task"], row["source"]["source_info"], row["source"]["prompt"])
    usage = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0}
    CURRENT_USAGE.set(usage)
    faithfulness, error = None, None
    async with semaphore:
        started = time.perf_counter()
        try:
            result = await metric.ascore(user_input=user_input, response=row["response"], retrieved_contexts=contexts)
            if result.value is None or math.isnan(result.value):
                error = "no_score"
            else:
                faithfulness = float(result.value)
        except Exception as exc:  # one failed response must not stop the run; it is recorded instead
            error = describe(exc)
        seconds = time.perf_counter() - started
    return {
        "id": row["id"],
        "split": row["split"],
        "task": row["task"],
        "model": row["model"],
        "label_any": bool(row["labels"]),
        "label_strict": any(not (label["implicit_true"] or label["due_to_null"]) for label in row["labels"]),
        "faithfulness": faithfulness,
        "error": error,
        "usage": usage,
        "seconds": round(seconds, 2),
    }


async def balance(http, settings):
    """Remaining DeepSeek balance in CNY, or None for other providers."""
    if "deepseek.com" not in settings["JUDGE_BASE_URL"]:
        return None
    reply = await http.get(
        settings["JUDGE_BASE_URL"].rstrip("/") + "/user/balance",
        headers={"Authorization": f"Bearer {settings['JUDGE_API_KEY']}"},
    )
    reply.raise_for_status()
    return sum(float(info["total_balance"]) for info in reply.json()["balance_infos"] if info["currency"] == "CNY")


async def run(pending, settings, concurrency):
    http = DefaultAsyncHttpxClient(timeout=300.0, event_hooks={"response": [record_usage]})
    async with http:
        client = AsyncOpenAI(
            api_key=settings["JUDGE_API_KEY"], base_url=settings["JUDGE_BASE_URL"], http_client=http, max_retries=4
        )
        metric = Faithfulness(llm=llm_factory(settings["JUDGE_MODEL"], client=client, max_tokens=MAX_TOKENS))
        before = await balance(http, settings)
        semaphore = asyncio.Semaphore(concurrency)
        RESULTS.mkdir(exist_ok=True)
        tasks = [asyncio.create_task(score(row, metric, semaphore)) for row in pending]
        failures = consecutive = 0
        with OUTPUT.open("a", encoding="utf-8") as handle:
            for count, finished in enumerate(asyncio.as_completed(tasks), 1):
                result = await finished
                failed = result["error"] not in (None, "no_score")
                failures += failed
                consecutive = consecutive + 1 if failed else 0
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                if count % 20 == 0 or count == len(pending):
                    print(f"{count}/{len(pending)} scored, {failures} failed", flush=True)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print(f"stopping after {consecutive} failures in a row (last: {result['error']})", flush=True)
                    break
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        after = await balance(http, settings)
    if before is not None:
        print(f"balance {before:.2f} -> {after:.2f} CNY, spent {before - after:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory holding response.jsonl and source_info.jsonl")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--limit", type=int, required=True, help="score at most this many new responses")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    settings = read_settings(SETTINGS)
    train, test = calibration_and_test(load_ragtruth(args.data))
    random.Random(20261110).shuffle(test)
    done = set()
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as handle:
            # A failed call is retried on the next run; analysis keeps the last record per response.
            done = {row["id"] for row in map(json.loads, handle) if row["error"] in (None, "no_score")}
    pending = [row for row in (train if args.split == "train" else test) if row["id"] not in done][: args.limit]
    print(f"{len(done)} already scored, scoring {len(pending)} {args.split} responses with {settings['JUDGE_MODEL']}")
    asyncio.run(run(pending, settings, args.concurrency))


if __name__ == "__main__":
    main()
