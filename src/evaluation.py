"""Rolling-origin backtest harness for the West nodal price model.

Shared evaluation used by every model-improvement experiment (baseline,
conformal intervals, IM re-tune, architecture bake-off) so candidates are
scored on the same volatile West holdout windows. Reports, per node and
aggregated:

- **CRPS** (probabilistic accuracy — the primary metric),
- **CI coverage** and **interval width** at a nominal interval (default the
  90% interval, 0.05–0.95),
- **MAE / RMSE / bias** of the median (point accuracy),
- **tail behavior**: median error and coverage conditioned on large-magnitude
  hours (``|actual| > tail_threshold``) and on negative-price hours.

Any Darts ``GlobalForecastingModel`` that supports ``historical_forecasts``
and probabilistic prediction works as the ``model`` argument — the served
``NaiveEnsembleModel``, a ``ConformalQRModel`` wrapper, a re-tuned TiDE, or a
foundation model.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mcrps, merr, mic, miw, rmse
from darts.models.forecasting.forecasting_model import ForecastingModel

# Put src/ on sys.path so the bare `import parameters` resolves regardless of
# how this library is imported (matches the other src/ modules).
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import parameters  # noqa: E402  (imported after the sys.path shim above)

log = logging.getLogger(__name__)


def backtest_report(
    model: ForecastingModel,
    series: list[TimeSeries],
    past_covariates: list[TimeSeries],
    future_covariates: list[TimeSeries],
    node_names: list[str] | None = None,
    forecast_horizon: int = parameters.FORECAST_HORIZON,
    holdout_days: int = 21,
    stride: int = 24,
    num_samples: int = 200,
    interval: tuple[float, float] = (0.05, 0.95),
    tail_threshold: float = 100.0,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Score a model with a rolling-origin backtest over a West holdout.

    For each node, forecasts are generated at daily origins across the last
    ``holdout_days`` of its series (each a full ``forecast_horizon`` ahead)
    and compared against the realized LMPs.

    Args:
        model: A predict-ready Darts model supporting ``historical_forecasts``
            and probabilistic prediction (e.g. the served ensemble).
        series: Per-node target LMP series (e.g. ``de.get_series(lmp_all)``).
        past_covariates: Per-node past covariates, aligned with ``series``.
        future_covariates: Per-node future covariates, aligned with ``series``.
        node_names: Optional per-node labels for the report index; defaults to
            each series' static-covariate node id, else ``node_0 … node_{n-1}``.
        forecast_horizon: Forecast length per origin, in hours.
        holdout_days: Width of the rolling-origin window at the series end;
            default 21 (~3 weeks of daily origins over the recent IM regime,
            enough windows to compare candidates without scoring on stale data).
        stride: Hours between successive origins (24 = daily).
        num_samples: Probabilistic samples per forecast; default 200 balances
            stable 0.05/0.95 tail quantiles against runtime (the served champion
            uses 500).
        interval: Nominal quantile interval scored for coverage/width.
        tail_threshold: ``|actual|`` above which an hour counts as a tail hour.

    Returns:
        A tuple of (per-node metrics DataFrame indexed by node name, aggregate
        metrics Series pooled across all windows and nodes, eval-metadata dict).
        Metric columns: ``crps, coverage, width, mae, rmse, bias, tail_mae,
        tail_coverage, neg_mae, neg_coverage, n_windows, n_tail, n_neg``. Note
        ``n_tail`` / ``n_neg`` count forecast-instance hours: with
        ``stride < forecast_horizon`` the rolling windows overlap, so a realized
        hour is counted once per forecast that covers it, not once overall. The
        eval-metadata dict records the exact scored window (``test_start`` /
        ``test_end`` — the realized-hour range) and the eval config
        (``holdout_days, stride, forecast_horizon, num_samples, interval,
        tail_threshold``), so a model's metrics carry the test set they were
        computed on.
    """
    if node_names is None:
        node_names = [
            str(s.static_covariates_values()[0][0]) if s.has_static_covariates
            else f'node_{i}'
            for i, s in enumerate(series)
        ]

    q_lo, q_hi = interval
    rows = []
    # Track the exact realized-hour range actually scored, across all nodes.
    win_start = None
    win_end = None
    for i, name in enumerate(node_names):
        # Series are hourly, so the rolling-origin window is expressed in hours:
        # back off holdout_days plus one horizon from the series end so the first
        # origin still has a full forecast_horizon of realized LMPs to score
        # against (with overlap_end=False no forecast runs past the series end).
        forecasts = model.historical_forecasts(
            series=series[i],
            past_covariates=past_covariates[i],
            future_covariates=future_covariates[i],
            start=series[i].end_time()
            - pd.Timedelta(hours=holdout_days * 24 + forecast_horizon),
            forecast_horizon=forecast_horizon,
            stride=stride,
            num_samples=num_samples,
            retrain=False,
            last_points_only=False,
            overlap_end=False,
            verbose=False,
        )
        if not forecasts:
            log.warning('no backtest windows for %s; skipping', name)
            continue

        actuals = [series[i].slice_intersect(f) for f in forecasts]
        # Widen the scored-window bounds to the realized hours this node covered.
        node_start = min(a.start_time() for a in actuals)
        node_end = max(a.end_time() for a in actuals)
        win_start = node_start if win_start is None else min(win_start, node_start)
        win_end = node_end if win_end is None else max(win_end, node_end)
        # Headline metrics via Darts (probabilistic + median-quantile point).
        row = {
            'node': name,
            'crps': _reduce(mcrps(actuals, forecasts)),
            'coverage': _reduce(mic(actuals, forecasts, q_interval=interval)),
            'width': _reduce(miw(actuals, forecasts, q_interval=interval)),
            'mae': _reduce(mae(actuals, forecasts, q=0.5)),
            'rmse': _reduce(rmse(actuals, forecasts, q=0.5)),
            # merr = mean(actual - median); positive => model under-forecasts.
            'bias': _reduce(merr(actuals, forecasts, q=0.5)),
            'n_windows': len(forecasts),
        }
        # Point-level arrays for tail conditioning (Darts metrics can't
        # condition on the actual value).
        av = np.concatenate([a.values().ravel() for a in actuals])
        lo = np.concatenate([f.quantile(q_lo).values().ravel() for f in forecasts])
        md = np.concatenate([f.quantile(0.5).values().ravel() for f in forecasts])
        hi = np.concatenate([f.quantile(q_hi).values().ravel() for f in forecasts])
        row.update(_tail_metrics(av, lo, md, hi, tail_threshold))
        rows.append(row)

    per_node = pd.DataFrame(rows).set_index('node')
    aggregate = _aggregate(per_node)
    _log_summary(per_node, aggregate, interval, tail_threshold)
    # Record exactly what/how was scored so metrics from different retrains can
    # be checked for comparability — the rolling holdout slides forward as the
    # series grows, so two models' numbers are only comparable on the same
    # test_start..test_end and eval config.
    eval_meta = {
        'test_start': str(win_start) if win_start is not None else None,
        'test_end': str(win_end) if win_end is not None else None,
        'holdout_days': holdout_days,
        'stride': stride,
        'forecast_horizon': forecast_horizon,
        'num_samples': num_samples,
        'interval': list(interval),
        'tail_threshold': tail_threshold,
    }
    return per_node, aggregate, eval_meta


def compare_candidate_to_champion(
    candidate_model: ForecastingModel,
    series: list[TimeSeries],
    past_covariates: list[TimeSeries],
    future_covariates: list[TimeSeries],
    target: str,
    num_samples: int = 100,
    nodes: list[str] | None = None,
) -> tuple[pd.Series, pd.Series | None, bool]:
    """Decide whether a freshly-trained candidate should replace the champion.

    Backtests the candidate and the target's current champion on the *same*
    recent window over a fixed node subset with reduced samples — a fast
    promote-gate (a relative CRPS ranking is robust to fewer nodes/samples, so
    this is much cheaper than the full harness). Both models must be scored
    together here because the rolling holdout slides as the series grows. The
    node subset is held constant (``node_list.EVAL_NODES``) so gate scores are
    comparable across retrains.

    Args:
        candidate_model: The just-trained ensemble under consideration.
        series: Per-node target series (full node list).
        past_covariates: Per-node past covariates, aligned with ``series``.
        future_covariates: Per-node future covariates, aligned with ``series``.
        target: Forecast target (parameters.TARGETS) whose champion to load.
        num_samples: Probabilistic samples per forecast (default 100).
        nodes: Node names to score; defaults to ``node_list.EVAL_NODES``.

    Returns:
        ``(candidate_aggregate, champion_aggregate, candidate_wins)``. The
        champion aggregate is ``None`` and ``candidate_wins`` is ``True`` when
        the target has no champion yet (nothing to beat).
    """
    import tempfile

    import node_list
    import utils
    from botocore.exceptions import ClientError
    from modeling import load_ensemble_from_dir

    gate_nodes = node_list.EVAL_NODES if nodes is None else nodes
    # Select the fixed gate nodes by their static-covariate id (order-independent),
    # so the same nodes are scored regardless of the series' ordering.
    idx = [
        i for i, ts in enumerate(series)
        if ts.has_static_covariates
        and str(ts.static_covariates_values()[0][0]) in gate_nodes
    ]
    if len(idx) < len(gate_nodes):
        log.warning('gate: %d/%d EVAL_NODES present in series', len(idx), len(gate_nodes))
    s = [series[i] for i in idx]
    p = [past_covariates[i] for i in idx]
    f = [future_covariates[i] for i in idx]
    _pn, cand_agg, _m = backtest_report(candidate_model, s, p, f, num_samples=num_samples)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            utils.download_champion_checkpoints(tmpdir, target=target)
        except ClientError as e:
            if e.response['Error']['Code'] in ('NoSuchKey', '404'):
                log.info('no current %s champion; candidate promotes by default', target)
                return cand_agg, None, True
            raise
        champ_model, _ts = load_ensemble_from_dir(tmpdir)

    _pn, champ_agg, _m = backtest_report(champ_model, s, p, f, num_samples=num_samples)
    return cand_agg, champ_agg, float(cand_agg['crps']) < float(champ_agg['crps'])


def _tail_metrics(
    actual: np.ndarray,
    lo: np.ndarray,
    median: np.ndarray,
    hi: np.ndarray,
    tail_threshold: float,
) -> dict[str, float]:
    """Median error and interval coverage on tail and negative-price hours."""
    abs_err = np.abs(actual - median)
    covered = (actual >= lo) & (actual <= hi)
    big = np.abs(actual) > tail_threshold
    neg = actual < 0
    return {
        'tail_mae': _masked_mean(abs_err, big),
        'tail_coverage': _masked_mean(covered, big),
        'neg_mae': _masked_mean(abs_err, neg),
        'neg_coverage': _masked_mean(covered, neg),
        'n_tail': int(big.sum()),
        'n_neg': int(neg.sum()),
    }


def _aggregate(per_node: pd.DataFrame) -> pd.Series:
    """Pool per-node metrics, each weighted by its own denominator.

    Whole-series metrics are weighted by ``n_windows`` (every window is one
    forecast_horizon long, so equal weight). The tail / negative-hour metrics
    are averages over *different* denominators, so they are weighted by
    ``n_tail`` / ``n_neg`` — not ``n_windows`` — and nodes with no tail/neg
    hours (whose per-node value is NaN) are dropped rather than poisoning the
    pooled number. Counts are summed.
    """
    agg = {}
    win = per_node['n_windows'].to_numpy()
    for m in ['crps', 'coverage', 'width', 'mae', 'rmse', 'bias']:
        agg[m] = _weighted(per_node[m].to_numpy(), win)
    n_tail = per_node['n_tail'].to_numpy()
    n_neg = per_node['n_neg'].to_numpy()
    agg['tail_mae'] = _weighted(per_node['tail_mae'].to_numpy(), n_tail)
    agg['tail_coverage'] = _weighted(per_node['tail_coverage'].to_numpy(), n_tail)
    agg['neg_mae'] = _weighted(per_node['neg_mae'].to_numpy(), n_neg)
    agg['neg_coverage'] = _weighted(per_node['neg_coverage'].to_numpy(), n_neg)
    for c in ['n_windows', 'n_tail', 'n_neg']:
        agg[c] = int(per_node[c].sum())
    return pd.Series(agg)


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean that ignores NaN values and zero/NaN weights.

    Returns NaN only when no entry has both a finite value and a positive
    weight (e.g. no node had any tail hour).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = ~np.isnan(values) & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else float('nan')


def _reduce(value: float | np.ndarray) -> float:
    """Collapse a Darts metric result (scalar or per-series array) to a float."""
    return float(np.nanmean(np.asarray(value)))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of ``values`` over ``mask``; NaN when the mask is empty."""
    return float(values[mask].mean()) if mask.any() else float('nan')


def _log_summary(
    per_node: pd.DataFrame,
    aggregate: pd.Series,
    interval: tuple[float, float],
    tail_threshold: float,
) -> None:
    """Log a readable per-node table and the aggregate row."""
    pct = int(round((interval[1] - interval[0]) * 100))
    log.info('backtest per node:\n%s', per_node.round(2).to_string())
    log.info(
        'AGGREGATE  crps=%.2f  cov%d=%.2f  width=%.1f  mae=%.2f  rmse=%.2f  '
        'bias=%.2f  tail_mae(|x|>%.0f)=%.2f  tail_cov=%.2f  neg_mae=%.2f  '
        'neg_cov=%.2f',
        aggregate['crps'], pct, aggregate['coverage'], aggregate['width'],
        aggregate['mae'], aggregate['rmse'], aggregate['bias'], tail_threshold,
        aggregate['tail_mae'], aggregate['tail_coverage'],
        aggregate['neg_mae'], aggregate['neg_coverage'],
    )
