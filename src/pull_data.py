"""Load the 20 Newsgroups test set that every condition is scored on."""

import json
from pathlib import Path
from typing import get_args

from sklearn.datasets import fetch_20newsgroups

from src.models.response_models import Label

SEED: int = 67
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


def load_test_set(limit: int | None = None) -> list[tuple[str, str]]:
    """Load the test split for the four labels, shuffled with SEED.

    Headers, footers and quotes are stripped so the label cannot leak
    from post metadata.

    Args:
        limit: Keep the first `limit` posts after shuffling; all if None.

    Returns:
        (text, label) pairs, where label is one of the `Label` values.
    """
    data: fetch_20newsgroups = fetch_20newsgroups(
        subset="test",
        categories=list(get_args(Label)),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=SEED,
    )

    pairs: list[tuple[str, str]] = [
        (text, data.target_names[i]) for text, i in zip(data.data, data.target)
    ]

    return pairs[:limit]


def save_test_set(
    pairs: list[tuple[str, str]], path: Path = DATA_DIR / "test.jsonl"
) -> Path:
    """Write (text, label) pairs to a JSON Lines file, one post per line.

    Args:
        pairs: (text, label) pairs, as returned by `load_test_set`.
        path: Output file; its directory is created if missing.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text, label in pairs:
            f.write(json.dumps({"text": text, "label": label}) + "\n")

    return path
