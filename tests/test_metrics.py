"""Tests for accuracy, macro-F1 and the bootstrap intervals."""

import pytest

from src.utils.metrics import accuracy, bootstrap_ci, macro_f1, paired_diff_ci

# 25 posts per class; every electronics post is predicted as space.
EXPECTED = (
    ["comp.graphics"] * 25
    + ["rec.sport.baseball"] * 25
    + ["sci.space"] * 25
    + ["sci.electronics"] * 25
)
PRED = EXPECTED[:75] + ["sci.space"] * 25


def test_macro_f1_is_lower_than_accuracy_when_a_class_collapses():
    """Accuracy hides the missing class; macro-F1 does not."""
    assert accuracy(EXPECTED, PRED) == 0.75
    # (1 + 1 + 2/3 + 0) / 4
    assert macro_f1(EXPECTED, PRED) == pytest.approx(2 / 3)


def test_none_prediction_is_a_miss_not_an_extra_class():
    """A refusal lowers recall without adding a fifth class."""
    pred = EXPECTED[:-1] + [None]
    # Electronics: TP 24, FN 1 -> F1 48/49; the rest are perfect.
    assert macro_f1(EXPECTED, pred) == pytest.approx((3 + 48 / 49) / 4)


def test_bootstrap_ci_contains_the_point_estimate():
    """The 95% interval brackets the full-sample value."""
    low, high = bootstrap_ci(EXPECTED, PRED, macro_f1)
    assert low < macro_f1(EXPECTED, PRED) < high


def test_paired_diff_ci_is_zero_for_identical_predictions():
    """Two identical conditions differ by exactly 0 on every resample."""
    assert paired_diff_ci(EXPECTED, PRED, PRED, macro_f1) == (0.0, 0.0)
