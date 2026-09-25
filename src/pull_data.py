"""Load the 20 Newsgroups test set that every condition is scored on."""

from typing import get_args

from sklearn.datasets import fetch_20newsgroups

from src.models.response_models import Label

SEED: int = 67


def load_test_set(limit: int | None = None) -> list[tuple[str, str]]:
    """Load the test split for the four labels, shuffled with SEED.

    Headers, footers and quotes are stripped so the label cannot leak
    from post metadata.

    Args:
        limit: Keep the first `limit` posts after shuffling; all if None.

    Returns:
        (text, label) pairs, where label is one of the `Label` values.
    """
    data = fetch_20newsgroups(
        subset="test",
        categories=list(get_args(Label)),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=SEED,
    )
    pairs = [
        (text, data.target_names[i])
        for text, i in zip(data.data, data.target)
    ]
    return pairs[:limit]
