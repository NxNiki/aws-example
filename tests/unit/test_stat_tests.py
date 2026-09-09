"""Unit tests for the numpy-only significance tests used by the Summary table.

Expected p-values are the reference values from scipy
(``ttest_ind(equal_var=False)`` / ``f_oneway``); the implementation must match
them without importing scipy.
"""

import math

import pytest

from bituslabs_ds.metrics.stat_tests import compare_groups, one_way_anova, welch_ttest

pytestmark = pytest.mark.unit


def test_welch_ttest_matches_scipy_reference():
    t, p = welch_ttest([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert t is not None and p is not None
    assert math.isclose(t, -1.8973665961010275, rel_tol=1e-9)
    assert math.isclose(p, 0.10753119493062718, rel_tol=1e-9)


def test_one_way_anova_matches_scipy_reference():
    f, p = one_way_anova([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert f is not None and p is not None
    assert math.isclose(f, 27.0, rel_tol=1e-9)
    assert math.isclose(p, 0.001, rel_tol=1e-9)


def test_undefined_when_too_few_values():
    assert welch_ttest([1.0], [2, 3, 4]) == (None, None)
    assert one_way_anova([[1, 2]]) == (None, None)


def test_undefined_when_no_variance():
    assert welch_ttest([5, 5, 5], [5, 5, 5]) == (None, None)
    assert one_way_anova([[5, 5], [5, 5], [5, 5]]) == (None, None)


def test_compare_groups_dispatch():
    a, b, c = [1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]
    assert compare_groups([a, b])[1] == "t-test"
    assert compare_groups([a, b, c])[1] == "anova"
    # Columns with <2 values don't count toward the test.
    assert compare_groups([a, [1.0]]) == (None, None)
    assert compare_groups([a]) == (None, None)
