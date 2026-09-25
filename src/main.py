"""Run the LLM vs embedding classification experiment.

Each post is classified by every condition in turn, one result line is appended
to data/results.jsonl per post, and a summary is printed.
"""

import argparse
import json
import logging
import pathlib
import textwrap
import time
from collections.abc import Callable
from statistics import mean, median

import httpx
import openai
from google.genai import errors as genai_errors

from src.clients.embed_client import EmbedClient
from src.clients.llm_clients import GeminiClient, GPTClient
from src.utils.metrics import accuracy, bootstrap_ci, macro_f1, paired_diff_ci
from src.models.candidate import Candidate
from src.models.response_models import LABEL_DESCRIPTIONS, Prediction
from src.utils.pull_data import DATA_DIR, TEST_PATH, read_test_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

# Prices: USD per 1M tokens, standard tier, prompts under 200k.
# https://developers.openai.com/api/docs/pricing
# https://ai.google.dev/gemini-api/docs/pricing
# Checked on 2026-09-24
GPT = Candidate(
    name="gpt",
    model_id="gpt-6-sol",
    input_usd_per_million_tok=2.00,
    output_usd_per_million_tok=10.00,
)
GEMINI = Candidate(
    name="gemini",
    model_id="gemini-3.1-pro-preview",
    input_usd_per_million_tok=2.00,
    output_usd_per_million_tok=12.00,
)
EMBED = Candidate(
    name="embed",
    model_id="gemini-embedding-2",
    input_usd_per_million_tok=0.20,
    output_usd_per_million_tok=0.00,
)
CANDIDATES: tuple[Candidate, ...] = (GPT, GEMINI, EMBED)

RESULTS_PATH: pathlib.Path = DATA_DIR / "results.jsonl"

# Failed call is recorded and the run continues, so one network blip does not
# throw away hours of paid calls.
API_ERRORS: tuple[type[Exception], ...] = (
    openai.APIError,
    genai_errors.APIError,
    httpx.HTTPError,
)

PROMPT: str = textwrap.dedent("""
    Classify the post below into exactly one of these topics.

    {labels}

    Post:
    {text}
    """)

LABEL_LIST: str = "\n".join(f"- {k}: {v}" for k, v in LABEL_DESCRIPTIONS.items())


def build_prompt(text: str) -> str:
    """Return the LLM prompt: instruction, labels with descriptions, post."""
    return PROMPT.format(labels=LABEL_LIST, text=text)


def timed(classify: Callable[[str], Prediction], text: str) -> dict:
    """Run one classification and record its label, tokens and latency."""
    start: float = time.perf_counter()

    try:
        pred: Prediction = classify(text)
        error: str | None = None
    except API_ERRORS as e:
        logger.warning("API error: %r", e)
        pred, error = Prediction(None, 0, 0), repr(e)

    latency_s: float = time.perf_counter() - start

    return {**pred._asdict(), "latency_s": latency_s, "error": error}


def run(pairs: list[tuple[str, str]]) -> list[dict]:
    """Classify every post with every condition, saving as it goes."""
    gpt: GPTClient = GPTClient(GPT.model_id)
    gemini: GeminiClient = GeminiClient(GEMINI.model_id)
    embed: EmbedClient = EmbedClient(EMBED.model_id)

    # Labels are embedded once; their one-off cost is not counted.
    label_vecs: dict[str, list[float]] = dict(
        zip(LABEL_DESCRIPTIONS, embed.embed(list(LABEL_DESCRIPTIONS.values())))
    )
    conditions: dict[str, Callable[[str], Prediction]] = {
        GPT.name: lambda t: gpt.classify(build_prompt(t)),
        GEMINI.name: lambda t: gemini.classify(build_prompt(t)),
        EMBED.name: lambda t: Prediction(embed.classify(t, label_vecs), 0, 0),
    }

    rows: list[dict] = []
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for i, (text, gold) in enumerate(pairs):
            row: dict = {"index": i, "gold": gold}

            for name, classify in conditions.items():
                row[name] = timed(classify, text)

            # Counted outside the timer so latency is the embed call alone.
            try:
                row[EMBED.name]["input_tokens"] = embed.count_tokens(text)
            except API_ERRORS as e:
                logger.warning("Token count failed: %r", e)

            f.write(json.dumps(row) + "\n")
            f.flush()

            rows.append(row)
            logger.info("Post %d/%d done.", i + 1, len(pairs))

    return rows


def summarize(rows: list[dict]) -> None:
    """Print metrics per condition and paired differences vs embedding."""

    gold: list[str] = [r["gold"] for r in rows]
    preds: dict[str, list[str]] = {
        c.name: [r[c.name]["label"] for r in rows] for c in CANDIDATES
    }

    logger.info(f"\nn = {len(rows)} posts")

    for c in CANDIDATES:
        results: list[dict] = [r[c.name] for r in rows]
        errors: int = sum(x["error"] is not None for x in results)

        # Failed calls would drag cost and latency toward 0.
        ok: list[dict] = [x for x in results if x["error"] is None] or results

        cost: float = mean(c.cost(x["input_tokens"], x["output_tokens"]) for x in ok)

        latencies: list[float] = [x["latency_s"] for x in ok]
        refusals: int = sum(x["label"] is None and x["error"] is None for x in results)

        logger.info(
            f"\n{c.name} ({c.model_id})\n"
            f"  Accuracy:     {_with_ci(gold, preds[c.name], accuracy)}\n"
            f"  Macro-F1:     {_with_ci(gold, preds[c.name], macro_f1)}\n"
            ""
            f"  Refusals:     {refusals}\n"
            f"  Errors:       {errors}\n"
            f"  Latency mean: {mean(latencies):.3f}s  "
            f"median {median(latencies):.3f}s\n"
            f"  cost per doc  ${cost:.6f}"
        )

    logger.info("\nmacro-F1 difference vs embed (95% paired bootstrap CI)")

    base = preds[EMBED.name]
    for c in (GPT, GEMINI):
        diff = macro_f1(gold, preds[c.name]) - macro_f1(gold, base)
        low, high = paired_diff_ci(gold, preds[c.name], base, macro_f1)

        logger.info(f"  {c.name} - embed  {diff:+.3f}  [{low:+.3f}, {high:+.3f}]")


def _with_ci(gold: list[str], pred: list, metric: Callable) -> str:
    low, high = bootstrap_ci(gold, pred, metric)
    return f"{metric(gold, pred):.3f}  [{low:.3f}, {high:.3f}]"


def main() -> None:
    """Load the test set, run every condition, and print the summary."""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="Score only the first N posts of data/test.jsonl.",
    )
    args: argparse.Namespace = parser.parse_args()

    # Fail before any API client is built, not mid-run.
    if not TEST_PATH.exists():
        parser.error(
            f"{TEST_PATH} not found; run `uv run python -m src.write_dataset`"
        )

    # The file is already shuffled and filtered, so results.jsonl line i
    # is test.jsonl line i.
    pairs: list[tuple[str, str]] = read_test_set(limit=args.limit)

    # Run the actual experiment and print the summary.
    summarize(run(pairs))


if __name__ == "__main__":
    main()
