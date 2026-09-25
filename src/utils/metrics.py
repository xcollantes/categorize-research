"""Accuracy, macro-F1 and bootstrap confidence intervals."""

import random
from collections import Counter
from collections.abc import Callable, Sequence
from statistics import quantiles
from typing import get_args

from src.models.response_models import Label
from src.utils.pull_data import SEED

LABELS: tuple[str, ...] = get_args(Label)
N_RESAMPLES: int = 1000

Gold = Sequence[str]
Preds = Sequence[str | None]
Metric = Callable[[Gold, Preds], float]


def accuracy(y_true: Gold, y_pred: Preds) -> float:
    """Return the fraction of predictions equal to the gold label."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def macro_f1(
    y_true: Gold, y_pred: Preds, labels: Sequence[str] = LABELS
) -> float:
    """Return the unweighted mean of per-class F1 over `labels`.

    A prediction outside `labels` (such as None for a refusal) counts as
    a miss for its gold class and never as an extra class.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, None where no label was returned.
        labels: The classes to average over.

    Returns:
        Macro-F1 in [0, 1].
    """
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    # 2TP / (2TP + FP + FN) equals the harmonic mean of P and R.
    return sum(
        2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) if tp[c] else 0.0
        for c in labels
    ) / len(labels)


def bootstrap_ci(
    y_true: Gold, y_pred: Preds, metric: Metric, seed: int = SEED
) -> tuple[float, float]:
    """Return the 95% percentile bootstrap interval of `metric`."""
    return _ci([
        metric(*_take(idx, y_true, y_pred))
        for idx in _resamples(len(y_true), seed)
    ])


def paired_diff_ci(
    y_true: Gold, pred_a: Preds, pred_b: Preds, metric: Metric,
    seed: int = SEED,
) -> tuple[float, float]:
    """Return the 95% bootstrap interval of metric(A) - metric(B).

    Both conditions are scored on the same resample each time, so the
    interval reflects their difference rather than two separate noises.
    If it contains 0, A and B are not distinguishable on this data.
    """
    diffs = []
    for idx in _resamples(len(y_true), seed):
        t, a, b = _take(idx, y_true, pred_a, pred_b)
        diffs.append(metric(t, a) - metric(t, b))

    return _ci(diffs)


def _resamples(size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [rng.choices(range(size), k=size) for _ in range(N_RESAMPLES)]


def _take(idx: list[int], *seqs: Sequence) -> list[list]:
    return [[s[i] for i in idx] for s in seqs]


def _ci(values: list[float]) -> tuple[float, float]:
    # n=40 gives cut points every 2.5%, so the first and last are the
    # 2.5th and 97.5th percentiles.
    cuts = quantiles(values, n=40, method="inclusive")
    return cuts[0], cuts[-1]
