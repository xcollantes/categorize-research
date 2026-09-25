"""Download the 20 Newsgroups test set and write it to data/test.jsonl.

Writes the exact posts the experiment scores (four labels, stripped,
empty posts dropped, shuffled with seed 67), one JSON object per line:
{"text": ..., "label": ...}. The download is cached by scikit-learn in
~/scikit_learn_data, so reruns are fast. No API keys are needed.

Usage, from the repo root:

    uv run python -m src.write_dataset
    uv run python -m src.write_dataset --out data/other.jsonl

Run it as a module (-m), not as `python src/write_dataset.py`; the
`src.` imports only resolve from the repo root.
"""

import argparse
import logging
from pathlib import Path

from src.utils.pull_data import TEST_PATH, load_test_set, save_test_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Pull the test set and save it to disk."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=TEST_PATH,
        help="Output file (default: data/test.jsonl).",
    )
    args: argparse.Namespace = parser.parse_args()

    pairs: list[tuple[str, str]] = load_test_set()
    path: Path = save_test_set(pairs, args.out)

    logger.info("Wrote %d posts to %s.", len(pairs), path)


if __name__ == "__main__":
    main()
