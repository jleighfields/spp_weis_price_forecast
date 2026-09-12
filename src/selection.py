"""Model-selection objectives — the single source of truth for how models rank.

Kept dependency-light (no sklearn/darts) so darts-free callers — the
`scripts/tune_parameters.py` CLI — can import the objective table without
pulling in the model stack. `src/parameters.py` re-exports `OBJECTIVES` /
`DEFAULT_OBJECTIVE` so darts-side callers can reach them the usual way.

An *objective mode* decides how a candidate model is ranked, at every point a
ranking happens: the Optuna study objective, the top-N bake into
`TIDE_PARAMS_<TARGET>`, and the champion/challenger promote gate. All three read
this table, so the metric a model is tuned on is the metric it is promoted on.

Each mode declares:
  `metrics`   — the Optuna objective tuple. Two entries means a two-objective
                study (`directions=['minimize', 'minimize']`); one means a
                single-objective study.
  `intervals` — the quantile band(s) whose coverage is scored, as
                ``(q_low, q_high)`` pairs. Empty for a mode that does not rank
                on calibration at all.
  `scalers`   — per-band weights, aligned 1:1 with `intervals`.

The rank score is::

    ci_err = sum over bands b of  scalers[b] * 100 * |coverage_b - nominal_b|
    score  = metrics[0] + ci_err

Per-band weights (rather than one weight over an average) exist because bands
are not equally trustworthy: coverage at the 95% band rests on far fewer
exceedances than coverage at the 50% band, so a weight can say so.

The weights are a *total*: the formula these modes restore was
``MAE + 0.5 * ci_err`` on a single band, so the day-ahead mode's two bands at
0.25 each keep the same total calibration pressure on the ranking while
estimating it from more of the predictive distribution.

Modes are per target because the weight is an exchange rate against that
target's own error scale — see ``mae_ci_rt`` below. The mode is part of the
Optuna study name, so switching targets or weights never mixes trials that were
ranked by different formulas.
"""

import os

# Bands whose coverage is measured on every trial and every backtest, in every
# mode — including modes that do not rank on calibration at all, so a model
# scored under one mode can be re-judged under another (or under different
# weights on these bands) without re-running anything. Must cover the union of
# every mode's `intervals`, which the tests enforce, and every edge must be a
# level in parameters.QUANTILES.
DIAGNOSTIC_BANDS = (
    (0.1, 0.9),    # 80%
    (0.05, 0.95),  # 90%
)

OBJECTIVES = {
    # Day-ahead: the restored formula, widened across two bands. The weights
    # sum to the 0.5 the original single-band `MAE + 0.5 * ci_err` carried.
    'mae_ci_da': {
        'metrics': ('mae', 'ci_err'),
        'intervals': ((0.1, 0.9), (0.05, 0.95)),
        'scalers': (0.25, 0.25),
    },
    # Real-time: same shape, heavier calibration weight. The weight is an
    # exchange rate — one percentage point of coverage error costs that many
    # $/MWh of MAE — so a fixed weight buys less influence as the target's
    # error scale grows. RT's MAE runs ~6x DA's (≈50 vs ≈8), so 0.25 there
    # would make calibration a near-tiebreaker. These are per-target rather
    # than one shared number for exactly that reason; never assume a weight
    # transfers between targets without checking their error scales.
    'mae_ci_rt': {
        'metrics': ('mae', 'ci_err'),
        'intervals': ((0.1, 0.9), (0.05, 0.95)),
        'scalers': (0.4, 0.4),
    },
    # CRPS as the accuracy term, same calibration treatment. MAE and CRPS are
    # the same order of magnitude on both targets, so the weights carry over
    # without rescaling — re-check that if either metric's scale shifts.
    'crps_ci': {
        'metrics': ('crps', 'ci_err'),
        'intervals': ((0.1, 0.9), (0.05, 0.95)),
        'scalers': (0.25, 0.25),
    },
    # single-objective CRPS — the behavior this table replaced, kept runnable as
    # a baseline. Empty tuples rather than a zero weight: there is no second
    # term to weight and no band to rank at, and a 0.0 invites someone to "fix"
    # it to a nonzero value and expect an effect.
    'crps': {
        'metrics': ('crps',),
        'intervals': (),
        'scalers': (),
    },
}

DEFAULT_OBJECTIVE = 'mae_ci_da'

# Env var name; read at the three call sites via mode_name_from_env().
OBJECTIVE_ENV_VAR = 'OBJECTIVE_MODE'

# How many top-ranked trials become ensemble members. Lives here rather than in
# parameters.py because the bake CLI — which writes exactly this many param
# dicts — cannot import parameters (it pulls in sklearn/darts). parameters.py
# re-exports it, so `parameters.TOP_N` still resolves. A second home would let
# the bake write N entries while the retrain slices a different N, silently
# training a smaller ensemble than the baked params claim.
TOP_N = 5


def study_name(model_name: str, model_type: str, mode_name: str) -> str:
    """Optuna study name for a (target, architecture, objective mode) triple.

    The one home for this format. The study notebook and the bake CLI must
    agree exactly: if they drift, the bake silently reads a different study
    than the sweep wrote, or fails to find one at all.

    Args:
        model_name: The target's model name, e.g.
            ``parameters.TARGETS[target]['model_name']``.
        model_type: Architecture key, e.g. ``'tide'``.
        mode_name: Objective mode key, e.g. ``'mae_ci_da'``.

    Returns:
        The study name, e.g. ``spp_west_da_tide_mae_ci_da``. The mode is part of it
        because trials scored under different objectives are not comparable —
        and because Optuna silently ignores a changed ``directions`` on an
        existing study rather than refusing it.
    """
    return f'{model_name}_{model_type}_{mode_name}'


def mode_name_from_env() -> str:
    """Return the active objective-mode name from the environment.

    Follows the same pattern as the ``TARGET`` env var: a sweep switches modes
    without editing tracked files.

    Returns:
        The value of ``OBJECTIVE_MODE``, or ``DEFAULT_OBJECTIVE`` if unset. Not
        validated here — pass it to ``resolve_mode`` for that.
    """
    return os.environ.get(OBJECTIVE_ENV_VAR, DEFAULT_OBJECTIVE)


def resolve_mode(name: str) -> dict:
    """Look up an objective mode by name, failing loudly on an unknown one.

    Args:
        name: An objective-mode key, e.g. ``'mae_ci_da'``.

    Returns:
        The mode's config dict from ``OBJECTIVES``.

    Raises:
        ValueError: If ``name`` is not a key of ``OBJECTIVES``. A typo must not
            silently fall back to the default — that would rank a study by an
            objective nobody chose.
    """
    if name not in OBJECTIVES:
        raise ValueError(
            f'unknown objective mode {name!r}; '
            f'valid modes are {sorted(OBJECTIVES)}'
        )
    return OBJECTIVES[name]


def band_nominal(interval: tuple[float, float]) -> float:
    """Nominal coverage of a quantile band, e.g. ``(0.1, 0.9)`` -> ``0.8``."""
    q_low, q_high = interval
    return round(q_high - q_low, 6)


def band_pct(interval: tuple[float, float]) -> int:
    """Nominal coverage as whole percent, e.g. ``(0.1, 0.9)`` -> ``80``."""
    return int(round(band_nominal(interval) * 100))


def band_label(interval: tuple[float, float]) -> str:
    """Short name for a band's coverage error, e.g. ``(0.1, 0.9)`` -> ``ci_err_80``.

    Used as the per-band metric name in backtests and as the Optuna
    ``user_attr`` name, so a trial's recorded diagnostics say which band each
    number is for.
    """
    return f'ci_err_{band_pct(interval)}'


def coverage_label(interval: tuple[float, float]) -> str:
    """Column name for a band's realized coverage, e.g. ``coverage_80``.

    The counterpart of ``band_label`` for the backtest report, which records
    realized coverage per band and derives the error from it.
    """
    return f'coverage_{band_pct(interval)}'


def coverage_error(coverage: float, interval: tuple[float, float]) -> float:
    """Percentage-point deviation of realized coverage from a band's nominal.

    Args:
        coverage: Realized coverage of ``interval``, as a fraction in [0, 1].
        interval: The ``(q_low, q_high)`` band ``coverage`` was measured on.

    Returns:
        ``100 * |coverage - nominal|`` — 0.0 for a perfectly calibrated band.
    """
    return 100.0 * abs(coverage - band_nominal(interval))


def weight_ci_errs(errors: dict[tuple[float, float], float], mode: dict) -> float:
    """Combine per-band coverage *errors* into the mode's calibration term.

    The single home for the weighting step. Callers already holding errors in
    percentage points — the Optuna study, which gets them straight out of
    ``backtest`` — use this; callers holding realized coverage use
    ``weighted_ci_err``, which converts and delegates here. Both paths must
    weight identically, or the study would optimize a different number than the
    promote gate decides on.

    Args:
        errors: Coverage error in percentage points, keyed by ``(q_low, q_high)``
            band. Must contain every band in ``mode['intervals']``; extra bands
            (the rest of ``DIAGNOSTIC_BANDS``) are ignored.
        mode: An ``OBJECTIVES`` entry.

    Returns:
        ``sum(scaler_b * error_b)`` over the mode's bands; 0.0 for a mode that
        does not rank on calibration.

    Raises:
        KeyError: If a band the mode ranks on is missing from ``errors``.
        ValueError: If the mode's ``intervals`` and ``scalers`` differ in
            length, which would pair a band with another band's weight.
    """
    return sum(
        scaler * errors[interval]
        for interval, scaler in zip(mode['intervals'], mode['scalers'], strict=True)
    )


def weighted_ci_err(coverages: dict[tuple[float, float], float], mode: dict) -> float:
    """Combine per-band realized *coverage* into the mode's calibration term.

    The coverage-side wrapper around ``weight_ci_errs``: converts each band's
    realized coverage to an error, then applies the same weighting.

    Args:
        coverages: Realized coverage keyed by ``(q_low, q_high)`` band. Must
            contain every band in ``mode['intervals']``; extra bands (the rest
            of ``DIAGNOSTIC_BANDS``) are ignored.
        mode: An ``OBJECTIVES`` entry.

    Returns:
        The weighted calibration term; 0.0 for a mode that does not rank on
        calibration.

    Raises:
        KeyError: If a band the mode ranks on is missing from ``coverages``.
    """
    errors = {
        interval: coverage_error(coverages[interval], interval)
        for interval in mode['intervals']
    }
    return weight_ci_errs(errors, mode)


def trial_metrics(
    values: tuple[float, ...] | list[float],
    user_attrs: dict,
    swept_mode: dict,
) -> dict[str, float]:
    """Every metric a trial recorded, by name, whatever its role in the sweep.

    A trial splits its metrics across two places: the ones its mode optimized
    are Optuna objective ``values``, the rest are ``user_attrs``. Re-ranking
    needs them in one namespace, so recombine.

    Args:
        values: The trial's objective values, in ``swept_mode['metrics']`` order.
        user_attrs: The trial's Optuna user attributes (non-numeric entries,
            e.g. ``model_path``, are dropped).
        swept_mode: The mode the study was *run* under — which is what says
            how to name ``values``.

    Returns:
        Metric name -> value, e.g. ``{'mae': .., 'crps': .., 'ci_err_80': ..}``.
    """
    metrics = {
        k: float(v) for k, v in user_attrs.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    metrics.update(zip(swept_mode['metrics'], (float(v) for v in values), strict=True))
    return metrics


def score_metrics(metrics: dict[str, float], mode: dict) -> float:
    """Score a recorded trial under any mode, recomputing the calibration term.

    The correct way to re-rank a finished study. ``selection_score`` sums a
    mode's own objective values and so assumes ``values[1]`` was *already*
    weighted by that mode — true during a sweep, false the moment you re-rank
    under different weights. This recomputes ``ci_err`` from the per-band
    errors every trial records, so the requested weights are actually applied.

    Args:
        metrics: A trial's metrics by name, from ``trial_metrics``.
        mode: The ``OBJECTIVES`` entry to rank under.

    Returns:
        The rank score under ``mode``, lower is better.

    Raises:
        KeyError: If the trial lacks a metric the mode needs — a per-band error
            for one of its bands, or its accuracy metric. Better to fail than
            to rank by whatever happens to be present.
    """
    values = []
    for name in mode['metrics']:
        if name == 'ci_err':
            errors = {b: metrics[band_label(b)] for b in mode['intervals']}
            values.append(weight_ci_errs(errors, mode))
        else:
            values.append(metrics[name])
    return selection_score(tuple(values), mode)


def selection_score(values: tuple[float, ...] | list[float], mode: dict) -> float:
    """Collapse a mode's objective values into one number, lower is better.

    The calibration term in ``values`` is already weighted (it comes from
    ``weighted_ci_err``), so this is a plain sum — the weighting lives in the
    mode table, not here.

    Args:
        values: The mode's objective values, in ``mode['metrics']`` order.
        mode: An ``OBJECTIVES`` entry.

    Returns:
        The rank score.

    Raises:
        ValueError: If ``values`` does not have one entry per metric — which
            means a study's objective count disagrees with the mode it is being
            ranked under, and the resulting order would be meaningless.
    """
    if len(values) != len(mode['metrics']):
        raise ValueError(
            f'expected {len(mode["metrics"])} objective value(s) for metrics '
            f'{mode["metrics"]}, got {len(values)}: {values}'
        )
    return float(sum(values))
