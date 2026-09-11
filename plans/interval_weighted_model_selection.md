# Interval-weighted model selection (revert CRPS-only ranking)

> **Status (2026-09-11): Phases 1-5 implemented; Phase 6 needs the GPU box.**
> Restores the two-objective Optuna study (MAE + CI coverage error) and the
> composite-ranked ensemble that preceded commit `64219c4` (2026-07-06, "Switch
> Optuna study to single-objective CRPS"), behind a selectable objective-mode
> flag.
>
> Code complete and verified offline: `src/selection.py` + `parameters` re-export,
> band-parametrized `get_ci_err`, mode-driven study/bake/gate, per-band coverage
> in `backtest_report`, `evaluation.score_aggregate`, docs. 222 unit tests pass
> (30 new), zero new ruff findings against the HEAD baseline, both notebooks
> clean under `marimo check`. The metric path was smoke-tested end to end on a
> synthetic `LinearRegressionModel`: all four bands score through Darts
> `backtest`, and all three modes produce sensible composites. Both silent-failure
> guards (T1 direction mismatch, T2 objective-count mismatch) were fired
> deliberately and exit with a named error.
>
> A `code-reviewer` pass then found four single-source-of-truth violations and
> one display bug, all fixed: `get_ci_err` and `_log_summary` had their own
> copies of the coverage-error / band-percent arithmetic; the study cell
> hand-wrote the band weighting (now `selection.weight_ci_errs`, shared with the
> gate); the Optuna study-name format lived in both the notebook and the bake CLI
> (now `selection.study_name`); and `plot_param_importances` silently rendered
> nothing in single-objective mode. `TOP_N` also moved into `selection.py` so the
> bake CLI reads the same value the retrain slices to — it could not import
> `parameters`. 233 unit tests pass.
>
> **Remaining (Phase 6 steps 3-6):** the real Optuna sweeps, the bake, the
> retrain, and the rank-stability comparison. Those need the GPU box and R2
> credentials.

## Goal

Rank models on **both** point accuracy and prediction-interval accuracy again,
and build the ensemble from that composite — instead of ranking on CRPS alone.
The scheme is selected by a flag (`OBJECTIVE_MODE`) defaulting to the restored
two-objective form, so the alternatives — including today's CRPS-only ranking —
stay runnable for comparison rather than being deleted.
Conformal prediction stays out of the codebase.

## Background — what actually changed

Commit `64219c4` replaced a two-objective study with a single CRPS objective.
Before it:

| Step | Before `64219c4` | Today |
|---|---|---|
| Optuna study | `directions=["minimize","minimize"]`, `target_names=["MAE","CI_ERROR"]` | `direction="minimize"` on CRPS |
| Trial objective | `backtest(metric=[mae, get_ci_err])` → `(mae, ci_err)` | `score_trial_crps` → CRPS, MAE as a `user_attr` |
| Ensemble top-N | `get_best_trials(ci_scaler=0.5)`, ranked by `MAE + 0.5×CI_ERROR` | `sorted(..., key=t.value)` — CRPS only |
| Promote gate | (did not exist) | `cand_crps < champ_crps` |

`CI_ERROR` is `modeling.get_ci_err` (`src/modeling.py:487`): `100 × |coverage − 0.8|`
on the 0.1/0.9 interval. **The helper still exists** — it is only called from the
dead-end `TEST_BUILD_BACKTEST` diagnostic cell (`model.py:373,382`), so the revert
has a live function to build on rather than a from-scratch rewrite.

CRPS is now the sole ranking metric in three places, and all three change here:
1. the study objective — `notebooks/model_training/model.py:392 score_trial_crps`
2. the top-N bake — `scripts/tune_parameters.py:87`
3. the promote gate — `src/evaluation.py compare_candidate_to_champion`

## Conformal — nothing to gate

**Conformal prediction was never merged.** It was prototyped in scratchpad only
and rejected twice: `a4d7129` ("evaluated out-of-sample, not adopting") and
`e23ecd5` ("57s/forecast latency + finicky config"). The measured result is in
`plans/completed/model_improvements_volatility.md` — coverage 0.85→0.88 but CRPS
19.6→20.1 from over-widening, and it did not fix the extreme tails.

No flag is added, because there is no code to gate. Writing a
`ConformalQRModel` wrapper now — for an approach already measured as worse on
the primary metric — would fail step 1 of the Minimalism checklist. The only
work here is deleting the two stale mentions that imply it exists:
`src/evaluation.py:4` and `:17`.

Worth recording, since it motivated this change: the "not enough data" instinct
is **correct about conformal specifically** — it needs a calibration split carved
out of an already-short holdout (IM data starts 2026-04-01; DA has no pre-launch
history at all). It does *not* transfer to CRPS. Coverage error is the **noisier**
of the two estimators on a short holdout — a binary hit-rate over a ~14-day test
window — so a coverage-weighted composite asks *more* of the data than CRPS
does, not less. That is a real cost of this change, accepted deliberately:
coverage is the quantity we want controlled, so we measure it directly even
though it is the noisier estimate. Phase 6 checks the rank stability this implies.

## Verified traps

Each of these was reproduced locally against the installed Darts 0.45.0 /
Optuna before writing this plan. They are the reason this is not a `git revert`.

### T1 — Optuna silently ignores a changed `directions` on an existing study
`create_study(directions=[...], load_if_exists=True)` against a study stored as
single-objective **does not raise**. It returns the study with its *stored*
single direction. Every trial then fails with
`The number of the values 2 did not match the number of the objectives 1`,
and `study.optimize` **does not propagate that** — it logs a warning per trial
and marks them `FAIL`. A 100-trial sweep would burn ~1.5 h and yield zero
usable trials, with a clean exit.

Both existing studies (`spp_west_tide`, `spp_west_da_tide`) are single-objective,
and the `/tune-parameters` skill currently documents `REMOVE_PRIOR_MODELS=False`
to resume — which walks straight into this.

**Fix:** assert the direction count immediately after `create_study`, and force a
fresh study on the first run (Phase 2).

### T2 — `trial.value` / `study.best_value` raise under multi-objective
Both raise `RuntimeError: This attribute is not available during multi-objective
optimization`. This breaks, on the first run after the change:
- `scripts/tune_parameters.py:87` (`key=lambda t: t.value`), `:92-94`, and
  `build_block`'s `trials[0].value` / `t.value` formatting;
- the `/tune-parameters` skill's monitoring one-liner (`round(s.best_value,3)`).

**Fix:** move both to `t.values` + the composite (Phases 3 and 5).

### T3 — `get_ci_err` returns garbage, silently, on non-list input
Given a single `TimeSeries` pair instead of lists, it iterates *timesteps*
rather than series and returns one value per hour, each a degenerate `20.0` or
`80.0`. It never errors.

Darts' `backtest` calls metrics once with flattened lists
(`metric_f(series_gen, forecasts_list, **kwargs)` in `ForecastingModel.backtest`),
so `metric=[mae, get_ci_err]` **does** work correctly on 0.45 — verified, as is
`pred.quantile([0.1, 0.9])` still producing `LMP_q0.100` / `LMP_q0.900` for the
column matching. But the function is one careless call away from silent nonsense.

**Fix:** a type guard that raises on non-sequence input (Phase 1).

## Fixes to carry along (deliberately *not* a literal revert)

- **Scoring window.** The old objective used `stride=25` with the default
  `last_points_only=True` — it scored only the **+120 h point** of each forecast,
  on a stride that drifts across hour-of-day. The current code uses `stride=24`
  and `last_points_only=False` (every hour of every horizon, daily origins),
  which matches both serving and the harness. **Keep the current window**; only
  the metric list changes.
- **`ci_scaler` has two disagreeing homes** — signature default `0.25`, call site
  `0.5` (`model.py`, pre-`64219c4`). That is precisely the "function default that
  silently disagrees with the config constant" anti-pattern in CLAUDE.md. It gets
  one home.
- **`get_ci_err` hardcodes `0.8` and the `'LMP'` column.** The nominal level moves
  to config; the target column is read off the merged frame instead.

## Design

### New leaf module `src/selection.py`

The composite is needed by the notebook (darts-side), by `src/evaluation.py`
(darts-side), and by `scripts/tune_parameters.py` — which is deliberately
**darts-free** (it imports `targets` directly with a comment saying so, because
`src/parameters.py` pulls in sklearn + darts at `parameters.py:8-9`). So the mode
table and the formula cannot live in `parameters.py`.

`src/selection.py` is a dependency-light leaf holding the `OBJECTIVES` table,
`DEFAULT_OBJECTIVE`, `selection_score`, and `resolve_mode` — mirroring the
existing `targets.py` ↔ `parameters.py` re-export pattern, with `parameters.py`
re-exporting for darts-side callers. Three real call sites (study, bake, gate),
so it clears Minimalism step 5.

The flag is the reason this is a table rather than a constant: a bare
`CI_SCALER` would have to be reinterpreted per mode (the weight means different
things against MAE and against CRPS), which is the duplicate-source-of-truth
drift CLAUDE.md warns about. Binding weight, interval, and metric names together
per mode keeps one home for all of it.

### `OBJECTIVE_MODE` — the selectable ranking schemes

Each mode declares its Optuna objective tuple, the band(s) whose coverage it
scores, and the weight on the calibration term. One table, in `selection.py`:

```python
OBJECTIVES = {
    # the restore, widened: MAE + per-band calibration on the 80% and 90% bands
    'mae_ci':  {'metrics': ('mae', 'ci_err'),
                'intervals': ((0.1, 0.9), (0.05, 0.95)), 'scalers': (0.25, 0.25)},
    # CRPS as the accuracy term, same calibration treatment
    'crps_ci': {'metrics': ('crps', 'ci_err'),
                'intervals': ((0.1, 0.9), (0.05, 0.95)), 'scalers': (0.25, 0.25)},
    # today's behavior, kept runnable as the baseline
    'crps':    {'metrics': ('crps',), 'intervals': (), 'scalers': ()},
}
DEFAULT_OBJECTIVE = 'mae_ci'
```

`intervals` and `scalers` are aligned tuples, so each band carries its own
weight:

```
ci_err = sum over bands b of  scalers[b] * 100 * |coverage_b - nominal_b|
score  = metrics[0] + ci_err
```

Per-band weights beat a single scaler with a mean, because bands are not equally
trustworthy: coverage at 98% rests on far fewer exceedances than coverage at
50%, and a per-band weight can say so. `crps` mode carries empty tuples — it
ranks on one metric, so there is no band to rank at and nothing to weight. Empty
tuples rather than `None`, so every consumer reads one field name of one type.

**The weights are a total, and the total is what was matched to the original.**
The formula being restored was `MAE + 0.5 x ci_err` on one band, so the two-band
weights sum to the same 0.5 rather than each carrying it. Against the DA
champion's recorded numbers (MAE 6.09, cov90 0.87 -> `ci_err` 3.0 points):

| weights | score | calibration share |
|---|---|---|
| `(0.5,)` on one band — the original | 6.09 + 1.5 = 7.59 | 20% |
| `(0.25, 0.25)` — as configured above | 6.09 + ~1.5 = 7.59 | 20% |
| `(0.5, 0.5)` | 6.09 + ~3.0 = 9.09 | 33% |

So this splits the original calibration pressure across two bands instead of
concentrating it on one — same total influence on the ranking, estimated from
more of the predictive distribution. Raising the total to `(0.5, 0.5)` is a
deliberate push beyond what the old formula did, and Phase 6 can test it as a
re-rank before anyone commits a sweep to it.

**One weight serves both accuracy terms.** MAE and CRPS are the same order of
magnitude on both targets — DA champion CRPS 4.43 / MAE 6.09; RT CRPS ~19.6 /
MAE 20.4-33.3 — so `crps_ci` does not need a rescaled weight relative to
`mae_ci`. (CRPS sits slightly below MAE on both, so the same weight gives
calibration modestly more influence in `crps_ci`: 3.0/4.43 vs 3.0/6.09 on DA.)

Band edges must be trained quantile levels. Verified against
`parameters.QUANTILES` (27 levels): the 50%, 80%, 90%, 95% and 98% bands all
have both edges in the set, so any of them can be listed without interpolation.

**Coverage is recorded at a fixed diagnostic set — 50/80/90/95 — in every mode**,
not merely at the bands the active mode ranks on. `ci_err` is a function of the
band, so a study that records only its own bands cannot be re-ranked under a
different weighting later, which would silently defeat the Phase 6 comparison.
All of them come off the same forecasts at negligible cost. This is also what
lets single-band rankings (the literal `MAE + 0.5 x ci_err_80` of the original)
be reconstructed offline without existing as named modes in the table.

### The mode goes in the study name — this kills T1

```python
study_name = f"{MODEL_NAME}_{MODEL_TYPE}_{OBJECTIVE_MODE}"
```

A mode switch then *cannot* load a stored study with an incompatible objective
count, which is the failure that silently burns a 1.5 h sweep. It also means the
existing `spp_west_tide` / `spp_west_da_tide` results stay readable as the
`crps`-mode baseline instead of being deleted to make room. The direction-count
assertion from T1 stays as a belt-and-braces guard against a hand-edited name.

> Note the one cost of the mode living in the study name: the existing studies
> are named without a suffix, so they are not auto-discovered as `crps`-mode
> studies. Either rename them once in sqlite, or accept that the `crps` baseline
> starts fresh.

### Bands: selection vs reporting

`get_ci_err` currently hardcodes the **80%** band; `backtest_report` defaults to
`interval=(0.05, 0.95)` (**90%**) for the app bands and `metrics.json`. Under the
mode table neither is a single global choice: selection scores the mode's own
bands, and the diagnostic set (50/80/90/95) is recorded alongside regardless.
`backtest_report` keeps its 90% default for the single headline `coverage` /
`width` columns the app and `metrics.json` already use, and gains per-band
coverage for the gate. This stays unambiguous rather than duplicated because
`backtest_report` already records the interval it used in `eval_meta`, so every
`metrics.json` says which band its headline coverage refers to.

### Where the flag is read

`OBJECTIVE_MODE` follows the existing `TARGET` pattern — env var with a config
default, so a sweep can switch modes without editing tracked files:

```python
OBJECTIVE_MODE = os.environ.get('OBJECTIVE_MODE', selection.DEFAULT_OBJECTIVE)
```

read in `model.py` (the study), `scripts/tune_parameters.py` (`--objective`, so
the bake is explicit about what it ranked on), and `model_retrain.py` (the gate).
The mode is recorded in `metrics.json` and in the baked `TIDE_PARAMS_<T>` header
comment, so no artifact is ambiguous about how it was chosen.

## Phases

### Phase 1 — single-home the scoring
- Add `src/selection.py`: the `OBJECTIVES` table, `DEFAULT_OBJECTIVE`,
  `selection_score(values, mode)`, and a `resolve_mode()` that validates an
  `OBJECTIVE_MODE` string against the table and fails with the valid names
  listed (a typo'd mode must not silently fall back to the default).
- Re-export `OBJECTIVES` / `DEFAULT_OBJECTIVE` from `src/parameters.py`, next to
  `TOP_N` (`parameters.py:37`), for darts-side callers.
- `src/modeling.py get_ci_err` (`:487`):
  - raise `TypeError` on non-sequence input (**T3**);
  - take the band as an argument instead of the hardcoded `0.8`/`0.1`/`0.9`,
    deriving the nominal level as `q_hi - q_lo` (so one function serves every
    band in the table);
  - take the actual column off the merged frame instead of the literal `'LMP'`;
  - update the docstring, which currently hardcodes "80%".
- Tests: `selection_score` for each mode, including the single-metric `crps`
  mode; `intervals` and `scalers` being equal length in every row (a misaligned
  pair would weight the wrong band); every band in the table having both edges in
  `parameters.QUANTILES`; `resolve_mode` rejecting an unknown name; `get_ci_err`
  on a known-coverage fixture at several bands; the `TypeError` guard.

### Phase 2 — mode-driven study (`notebooks/model_training/model.py`)
- Read `OBJECTIVE_MODE` from the env; derive `study_name` with the mode suffix.
- `create_study(directions=[...], ...)` — one `"minimize"` per metric in the
  mode; `target_names` from the mode's metric names.
- **Guard for T1**, right after `create_study`: raise unless
  `len(study.directions) == len(mode['metrics'])`, naming
  `REMOVE_PRIOR_MODELS=True` as the fix. The mode-suffixed study name should
  make this unreachable; it stays because the failure it catches is silent.
- Replace `score_trial_crps` with a mode-driven `score_trial` that runs **one**
  backtest with `stride=24`, `last_points_only=False`, `num_samples=200` and
  `metric=[mae, mcrps, *ci_err_per_band]` — one `get_ci_err` partial per band in
  the fixed 50/80/90/95 diagnostic set, all over the same forecasts — then
  returns the tuple the active mode asks for (its own bands, weighted and
  summed). Keep the `float('inf')` NaN guards. Every metric the mode does *not*
  optimize is stored as a `user_attr`, so **every trial carries MAE, CRPS, and
  coverage error at all four bands regardless of mode** — that, not the mode
  table, is what makes the Phase 6 re-ranking valid.
- Restore `get_best_trials` ranking on `selection_score(t.values, mode)` (no
  local `ci_scaler` argument — it reads the mode table).
- Restore the two Pareto cells + `plot_pareto_front` import, and the
  per-objective loops in the optimization-history / contour cells — all guarded
  to no-op in a single-objective mode, where `plot_pareto_front` is invalid.
- Restore the three-way `print_callback` (best accuracy / best CI / best
  composite), likewise degrading to the single-value form in `crps` mode.
- Delete the now-redundant `TEST_BUILD_BACKTEST` duplicate scoring block, whose
  only remaining job was exercising `get_ci_err` by hand.

### Phase 3 — composite bake (`scripts/tune_parameters.py`)
- Import `selection` alongside `targets` (both darts-free).
- Add `--objective` (default `selection.DEFAULT_OBJECTIVE`); it selects both the
  study name and the ranking, so a bake can never rank one mode's trials by
  another mode's formula.
- Rank on `selection_score(t.values, mode)`; replace every `t.value` (**T2**).
- `build_block`: header and per-trial comments become
  `# trial #N  MAE x  CI_ERR y  score z`, so a baked `TIDE_PARAMS_<T>` block says
  what it was ranked on. Existing blocks in `parameters.py:175,253` say "by CRPS"
  and are rewritten on the next bake.
- Guard: fail with a clear message if the loaded study's objective count does
  not match the requested mode (a mismatch would otherwise `RuntimeError` deep
  in a comprehension, per **T2**).

### Phase 4 — promote gate (`src/evaluation.py`)
- `compare_candidate_to_champion`: take a `mode` argument (default
  `DEFAULT_OBJECTIVE`), score with that mode's `interval`, and decide on
  `selection_score(...)` over the mode's metrics instead of `cand_crps < champ_crps`.
  `ci_err` is the weighted sum of `100*abs(coverage_b - nominal_b)` over the
  mode's bands, which needs `backtest_report` to return coverage per band rather
  than for one interval. When `intervals` is empty (`crps` mode), coverage is
  still reported at the diagnostic bands — just not ranked on.
- Keep the NaN → keep-champion behavior and its warning; extend it to the
  composite.
- `backtest_report` already returns `mae` and `coverage`, so no new metric math —
  do **not** fork the calculation.
- Retrain notebook (`model_retrain.py:484`): read `OBJECTIVE_MODE` from the env
  and pass it to the gate; the log line and `metrics.json`'s `primary_metric`
  (`:543`) name the active mode and its composite. CRPS stays in the metrics dict
  as a diagnostic in every mode — it is still the better-estimated number, and
  keeping it is what lets Phase 6 compare modes.
- **Gate comparability:** a champion promoted under one mode and a challenger
  scored under another are not comparable. The gate scores both models itself on
  the same window, so the *scores* are consistent — but record the mode in
  `metrics.json` so a mode switch is visible in the history rather than looking
  like a sudden metric jump.
- Delete the stale conformal mentions at `evaluation.py:4,17`.

### Phase 5 — docs
- `.claude/skills/tune-parameters/SKILL.md`: the monitor one-liner (`best_value`
  → `best_trials` + composite, **T2**); the step-3 gate description ("only
  promotes if the candidate wins on CRPS"); the "resumable study" note (**T1** —
  resumable only within one mode); the `OBJECTIVE_MODE` env var and
  `--objective` flag; and the mode-suffixed study names in the monitor command.
  Its `argument-hint` gains the mode.
- `CLAUDE.md`: `OBJECTIVE_MODE` and the `OBJECTIVES` table belong in the
  "single source of truth for parameter values" list, alongside `TARGETS`.
- `src/README.md` if it names the selection metric.

### Phase 6 — re-run and validate
1. ✅ `uv run ruff check` + `uv run pytest tests/unit -q` — 222 pass (30 new),
   no new lint findings vs the HEAD baseline.
2. ✅ **Smoke the objective** — done offline against a synthetic
   `LinearRegressionModel` rather than by burning trials: the four band metrics
   score through Darts `backtest` (6 metrics/row), all three modes produce
   finite composites, and `backtest_report` -> `score_aggregate` yields per-band
   `coverage_50/80/90/95` and a usable gate score. Both guards fired on purpose.
   Still worth one `NUM_TRIALS=2` run on real data before the full sweep, to
   confirm the metric list survives a real TiDE and the study path end to end.
3. Full study per target on the GPU box (~100 trials, ~1.5 h each) in the
   default `mae_ci` mode. `da` first — it is the primary target.
4. Bake, retrain, let the gate decide. Record the composite **and** CRPS **and**
   coverage for champion and challenger.
5. **Rank-stability check** (the cost flagged above): every trial carries MAE,
   CRPS and coverage error at all four diagnostic bands, so re-rank the *same*
   study under any weighting — no re-running. Compare the top-5 sets under
   `(0.5, 0.5)`, `(0.25, 0.25)`, and the original single-band
   `MAE + 0.5 x ci_err_80`. (Valid only because Phase 2 records the fixed band
   set; recording just the active mode's bands would make a re-rank silently
   wrong.) If the two-band orderings are steadier than the single-band one, that
   is the evidence that the extra band is buying variance reduction rather than
   just extra weight — and the basis for settling the totals question above.
   Heavy disagreement with near-identical composite scores means coverage noise
   is driving selection on this holdout, in which case lower the mode's `scaler`
   rather than abandoning the approach. This is the payoff of recording the
   unused metrics in Phase 2: choosing between the four objectives costs one
   sweep, not four.
6. Only run a second full sweep in another mode if step 5 shows the re-ranking is
   not enough — i.e. the modes disagree about which *region of the search space*
   to explore, not just how to rank what was already explored.

## Rollback

Nothing is destructive except the Optuna studies, which are rebuilt by re-running.
Reverting is: restore `direction="minimize"` + `score_trial_crps`, point
`tune_parameters.py` back at `t.value`, restore the CRPS gate, re-run the sweeps.
The champion itself is unaffected — `champion.json` still points at whatever last
won, and `scripts/r2_promote_champion.py <ts> --target <t> --promote` repoints it
either way.
