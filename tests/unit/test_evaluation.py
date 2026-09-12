"""Unit tests for the numeric helpers in evaluation (the backtest harness).

The full ``backtest_report`` needs a fitted model and data, so these cover the
pure reduction/aggregation helpers it is built from.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import evaluation
import selection


class TestMaskedMean:
    def test_masked_mean_selects_masked_values(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        assert evaluation._masked_mean(vals, mask) == 2.0  # mean(1, 3)

    def test_masked_mean_empty_mask_is_nan(self):
        vals = np.array([1.0, 2.0])
        mask = np.array([False, False])
        assert math.isnan(evaluation._masked_mean(vals, mask))


class TestReduce:
    def test_reduce_scalar(self):
        assert evaluation._reduce(3.5) == 3.5

    def test_reduce_array_is_mean(self):
        assert evaluation._reduce(np.array([2.0, 4.0])) == 3.0

    def test_reduce_ignores_nan(self):
        assert evaluation._reduce(np.array([2.0, np.nan, 4.0])) == 3.0


class TestTailMetrics:
    def test_conditions_on_magnitude_and_sign(self):
        # actuals: one big spike (500), one negative (-50), two ordinary.
        actual = np.array([500.0, -50.0, 10.0, 20.0])
        median = np.array([400.0, -30.0, 12.0, 25.0])
        lo = np.array([450.0, -60.0, 0.0, 0.0])   # spike falls BELOW its lo
        hi = np.array([600.0, -10.0, 50.0, 50.0])
        out = evaluation._tail_metrics(actual, lo, median, hi, tail_threshold=100.0)

        assert out['n_tail'] == 1                 # only |500| > 100
        assert out['n_neg'] == 1                  # only -50 < 0
        assert out['tail_mae'] == 100.0           # |500 - 400|
        # spike actual 500 is not within [450, 600]? 500 is within -> covered.
        assert out['tail_coverage'] == 1.0
        assert out['neg_mae'] == 20.0             # |-50 - (-30)|
        assert out['neg_coverage'] == 1.0         # -50 within [-60, -10]

    def test_uncovered_tail(self):
        actual = np.array([500.0])
        median = np.array([300.0])
        lo = np.array([100.0])
        hi = np.array([400.0])                     # 500 > 400 -> not covered
        out = evaluation._tail_metrics(actual, lo, median, hi, tail_threshold=100.0)
        assert out['tail_coverage'] == 0.0
        assert out['tail_mae'] == 200.0


class TestAggregate:
    def test_window_weighted_means_and_count_sums(self):
        per_node = pd.DataFrame({
            'crps': [10.0, 20.0],
            'coverage': [0.8, 1.0],
            'width': [100.0, 200.0],
            'mae': [5.0, 15.0],
            'rmse': [6.0, 16.0],
            'bias': [-1.0, 1.0],
            'tail_mae': [50.0, 150.0],
            'tail_coverage': [0.5, 0.9],
            'neg_mae': [4.0, 8.0],
            'neg_coverage': [1.0, 1.0],
            'coverage_80': [0.7, 0.9],
            'coverage_90': [0.85, 0.95],
            'n_windows': [1, 3],
            'n_tail': [2, 4],
            'n_neg': [10, 20],
        }, index=['a', 'b'])
        agg = evaluation._aggregate(per_node)

        # crps weighted by n_windows: (10*1 + 20*3) / 4 = 17.5
        assert agg['crps'] == 17.5
        assert agg['coverage'] == (0.8 * 1 + 1.0 * 3) / 4
        # per-band coverage is pooled the same way as the headline column
        assert agg['coverage_80'] == (0.7 * 1 + 0.9 * 3) / 4
        assert agg['coverage_90'] == (0.85 * 1 + 0.95 * 3) / 4
        # tail_mae weighted by n_tail, NOT n_windows: (50*2 + 150*4) / 6
        assert agg['tail_mae'] == (50.0 * 2 + 150.0 * 4) / 6
        # neg_mae weighted by n_neg: (4*10 + 8*20) / 30
        assert agg['neg_mae'] == (4.0 * 10 + 8.0 * 20) / 30
        assert agg['n_windows'] == 4
        assert agg['n_tail'] == 6
        assert agg['n_neg'] == 30

    def test_node_with_no_tail_hours_does_not_poison_aggregate(self):
        # Node 'b' has zero tail hours, so its tail_mae is NaN — it must be
        # dropped from the pooled tail metric, not turn the aggregate to NaN.
        per_node = pd.DataFrame({
            'crps': [10.0, 20.0],
            'coverage': [0.9, 0.9],
            'width': [100.0, 100.0],
            'mae': [5.0, 5.0],
            'rmse': [6.0, 6.0],
            'bias': [0.0, 0.0],
            'tail_mae': [120.0, float('nan')],
            'tail_coverage': [0.8, float('nan')],
            'neg_mae': [5.0, 7.0],
            'neg_coverage': [1.0, 1.0],
            'n_windows': [2, 2],
            'n_tail': [4, 0],
            'n_neg': [10, 10],
        }, index=['a', 'b'])
        agg = evaluation._aggregate(per_node)

        assert agg['tail_mae'] == 120.0          # only node 'a' contributes
        assert not math.isnan(agg['tail_mae'])
        assert agg['n_tail'] == 4


class TestWeighted:
    def test_ignores_nan_values_and_zero_weights(self):
        vals = np.array([10.0, float('nan'), 30.0])
        weights = np.array([1.0, 5.0, 0.0])   # nan dropped, zero-weight dropped
        assert evaluation._weighted(vals, weights) == 10.0

    def test_all_invalid_is_nan(self):
        assert math.isnan(evaluation._weighted(np.array([float('nan')]), np.array([1.0])))
        assert math.isnan(evaluation._weighted(np.array([5.0]), np.array([0.0])))


class TestScoreAggregate:
    """Turning a backtest aggregate into the rank score the promote gate uses."""

    @staticmethod
    def _agg(**overrides):
        base = {
            'crps': 4.0,
            'mae': 6.0,
            'coverage_50': 0.5,
            'coverage_80': 0.8,
            'coverage_90': 0.9,
            'coverage_95': 0.95,
        }
        base.update(overrides)
        return pd.Series(base)

    def test_perfect_calibration_scores_the_accuracy_term_alone(self):
        mode = selection.resolve_mode('mae_ci_da')
        assert evaluation.score_aggregate(self._agg(), mode) == pytest.approx(6.0)

    def test_miscalibration_adds_weighted_penalty(self):
        mode = selection.resolve_mode('mae_ci_da')
        # 5 points off at 80%, 3 off at 90%, both weighted 0.25 -> +2.0
        agg = self._agg(coverage_80=0.75, coverage_90=0.87)
        assert evaluation.score_aggregate(agg, mode) == pytest.approx(8.0)

    def test_single_metric_mode_ignores_coverage(self):
        mode = selection.resolve_mode('crps')
        agg = self._agg(coverage_80=0.2, coverage_90=0.2)
        assert evaluation.score_aggregate(agg, mode) == pytest.approx(4.0)

    def test_crps_mode_uses_crps_not_mae(self):
        agg = self._agg()
        assert evaluation.score_aggregate(
            agg, selection.resolve_mode('crps_ci')
        ) == pytest.approx(4.0)
        assert evaluation.score_aggregate(
            agg, selection.resolve_mode('mae_ci_da')
        ) == pytest.approx(6.0)

    def test_nan_term_yields_nan_not_a_winning_score(self):
        # A scoring failure must never look like a great score, or the gate
        # would promote a model it could not evaluate.
        mode = selection.resolve_mode('mae_ci_da')
        assert np.isnan(evaluation.score_aggregate(self._agg(mae=np.nan), mode))
        assert np.isnan(
            evaluation.score_aggregate(self._agg(coverage_80=np.nan), mode)
        )

    def test_missing_ranked_band_raises(self):
        mode = selection.resolve_mode('mae_ci_da')
        agg = self._agg().drop('coverage_90')
        with pytest.raises(KeyError):
            evaluation.score_aggregate(agg, mode)
