"""Invariant tests for src/parameters.py constants.

QUANTILES is the single home for the QuantileRegression quantile set, and
three consumers select specific levels by exact float lookup — the eval
harness (0.05/0.5/0.95), get_ci_err (0.1/0.9), and the app/plotting
(0.1/0.5/0.9). These guard against an edit that drops, reorders, or drifts a
required level.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import parameters


class TestQuantiles:
    def test_strictly_ascending_no_duplicates(self):
        q = parameters.QUANTILES
        assert q == sorted(q)
        assert len(q) == len(set(q))

    def test_within_open_unit_interval(self):
        assert all(0.0 < x < 1.0 for x in parameters.QUANTILES)

    def test_contains_levels_downstream_code_selects(self):
        # exact-float lookups in the harness / get_ci_err / plotting
        for level in (0.05, 0.1, 0.5, 0.9, 0.95):
            assert level in parameters.QUANTILES

    def test_symmetric_about_median(self):
        q = set(parameters.QUANTILES)
        assert all(round(1 - x, 6) in q for x in q)
