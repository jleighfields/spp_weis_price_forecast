# Day-Ahead (DA) forecast framework alongside real-time (RT)

Add a **day-ahead LMP** forecasting model that runs in parallel to the existing
real-time (RT) model, reusing the same pipeline (TiDE ensemble, covariates,
eval harness, champion machinery). Goal: keep both frameworks so the RT model
can be revisited when the market matures, and demonstrate the process on the
much-more-forecastable DA target.

> Why DA: on the same West node / window, DA is far less random than RT —
> autocorr lag-1 0.86 vs 0.22, excess kurtosis 0.18 vs 160, and a naive
> persistence forecast is ~4× more accurate. RT's spikiness is a market
> property, not a model failure.

## Key design insight — the target switch lives at the data layer

`data_engineering.prep_lmp` reads a DuckDB `lmp` table keyed on the hour-ending
columns (`timestamp_mst_HE`, `GMTIntervalEnd_HE`, `Interval_HE`) and emits the
canonical `unique_id / timestamp_mst / LMP / lmp_diff` frame. **DA is already
hourly**, so its `Interval / GMTIntervalEnd / timestamp_mst` columns *are* the
hour-ending values. If `create_database` loads the DA parquet and aliases those
to the `*_HE` names, the `lmp` table is schema-identical to the RT one — and
**every downstream step (features, covariates, TimeSeries, model, backtest) is
reused unchanged.** The whole target switch is one parameter at the source.

## What is shared vs target-specific

**Shared (no change):** `FUTR_COLS` (MTLF, wind/solar forecasts, re_ratio, …
all valid future covariates for DA), `PAST_COLS` (lmp_diff/rolling auto-compute
on whichever LMP is the target; `Averaged_Actual` is shared load), the MTLF/MTRF
prep + joins, `modeling.py` (TiDE build/fit), `QUANTILES`, `evaluation.py`
backtest harness, `plotting.py`.

**Target-specific (parametrize):** the source parquet (`im/lmp.parquet` vs
`im/da_lmp.parquet`), `MODEL_NAME` (`spp_west` vs `spp_west_da`), the champion /
retrains storage namespace, and which model the app serves.

## Why two models, not one bivariate model

A single model predicting a two-component target `[RT_LMP, DA_LMP]` per node is
technically supported (Darts `TiDEModel` + `QuantileRegression` handle
multivariate targets). It was considered and **deferred** in favor of two
separate univariate models, because for this project's goal separate is better:

- **Lifecycle coupling.** The explicit goal is to park RT and revisit it when
  the market settles. A joint model can't retrain/re-champion/retire one target
  without touching the other; separate models have independent lifecycles.
- **Loss domination.** RT has ~5× the std and excess kurtosis 160 vs DA's 0.18.
  In one shared quantile loss, RT's variance and spikes would dominate the
  gradient and risk degrading DA calibration — which is the whole reason to
  forecast DA.
- **Per-target tuning.** Smooth DA and spiky RT want different regularization
  (epochs, dropout, input length); one architecture can't tune both.
- **Data plumbing.** A bivariate target needs RT and DA aligned on a common
  hourly index per node with missing-value handling (DA is lag-published/sparser,
  RT continuous). The chosen design keeps each target univariate and reuses the
  pipeline as-is.

Both approaches are actually a single *global* model trained across all nodes —
the question is univariate-target ×2 vs bivariate-target ×1, not "2 models vs 1".
A joint model is the right tool if the aim shifts to capturing the **DA→RT
basis** or multi-task gains; noted as a future experiment, not the demo path.

## Phase 1 — Target config (single home) — ✅ DONE

- `src/parameters.py`: `DEFAULT_TARGET='da'` (DA is the primary/demo model; RT
  parked) + `TARGETS` map (`rt`→{`source_dataset` `lmp`, `model_name` `spp_west`},
  `da`→{`da_lmp`, `spp_west_da`}); `MODEL_NAME` derived from the default target
  (now `spp_west_da`).
- `src/utils.py`: `retrains_prefix(target)`/`champion_key_suffix(target)` →
  `models/{target}/…`; the `RETRAINS_PREFIX`/`CHAMPION_KEY_SUFFIX` constants are
  the RT-default values (now `models/rt/…`) so existing callers are unchanged;
  `download_champion_checkpoints(dest, target='rt')`.
- Tests updated + added (target namespace, TARGETS invariants); 180 pass.
- **Deferred to deploy:** the RT champion storage migration `models/champion.json`
  + `models/retrains/*` → `models/rt/*` (Decision A) happens when the new code is
  deployed, not now — the live app still reads the old paths until then.

## Phase 2 — DA data path — ✅ DONE

- `create_database(datasets, target=None)` (default `parameters.DEFAULT_TARGET`):
  the price table is always named `lmp` but loaded from the target's source
  parquet. For `da` it uses DuckDB `SELECT * RENAME (Interval AS Interval_HE,
  GMTIntervalEnd AS GMTIntervalEnd_HE, timestamp_mst AS timestamp_mst_HE)` —
  RENAME (not alias) so no duplicate columns — making the `lmp` table
  schema-identical to RT. `prep_lmp` and everything after are unchanged.
- Verified against real `spp-rto` data: the RT and DA `lmp` tables have the
  identical 13-column schema, and both reduce through `prep_lmp` to the canonical
  `{unique_id, timestamp_mst, LMP, lmp_diff}` frame. DA's one-row-per-hour makes
  the `group_by(...).mean()` a no-op; `clip_outliers` stays False.
- Unit tests: RT reads `im/lmp.parquet` without RENAME; DA reads
  `im/da_lmp.parquet` with the `*_HE` RENAME (182 pass).

## Phase 3 — Training both targets — ✅ DONE (DA champion trained + promoted)

- `notebooks/model_training/model_retrain.py` parametrized by target (env
  `TARGET`, default `parameters.DEFAULT_TARGET`): `create_database(target=…)`,
  `MODEL_NAME` from `TARGETS`, uploads to `utils.retrains_prefix(TARGET)`,
  champion to `utils.champion_key_suffix(TARGET)`.
- `scripts/r2_promote_champion.py` gained `--target` (default `da`); its
  helpers thread the target's retrains prefix / champion key.
- First DA champion trained on the GPU box reusing RT `TIDE_PARAMS` (Decision D)
  and promoted: `models/da/champion.json` → `models/da/retrains/2026-07-10_14-43-38/`
  (5 TiDE models). Isolated from the live RT champion at `models/champion.json`.
- **Backtest (harness): CRPS 4.43, cov90 0.87, MAE 6.09, bias −1.72** — vs the
  RT champion's ~17.2 CRPS, a ~4× improvement, confirming the data analysis.
- `download_champion_checkpoints(target='da')` load path verified end-to-end.
- Deferred (Decision D): DA-specific Optuna re-tune → parametrize `model.py`
  then (not needed to reuse RT params now).

## Phase 4 — Serving (app)

- `app.py`: load both champions (RT + DA) and both target series, add a UI
  control to switch **Real-time ↔ Day-ahead** (reuses the node dropdown — both
  cover the same 64 West nodes — and `plotting.py`). **Decision B**: toggle vs
  side-by-side.
- Note the DA horizon semantics in the UI copy: a DA forecast is a prediction of
  future days' cleared prices (useful before each day's auction clears); the
  rolling model still emits a 120 h horizon.

## Phase 5 — Modal retrain job

- Parametrize the retrain by target. **Decision C**: one job file with two
  scheduled functions (`retrain_rt_weekly`, `retrain_da_weekly`, each with
  `env={"TARGET": ...}`) vs a single env-param'd job. Collection is unchanged —
  DA is already gathered into `im/da_lmp.parquet`.

## Phase 6 — Tests + docs

- Unit tests for the target config + DA `create_database` branch.
- Update `src/README.md`, `CLAUDE.md` (note the `target` dimension in the model
  config + champion layout), and `.env`/deploy notes if a `TARGET` env is added.

## Open decisions

- **A. Champion namespace.** Namespace both (`models/rt/`, `models/da/`) —
  cleaner, but migrates the current `models/champion.json` + `models/retrains/`
  into `models/rt/` (a small R2 move + one pointer rewrite). *Alternative:* leave
  RT at `models/champion.json` and only add `models/da/…` (no migration,
  asymmetric). **Recommend: namespace both.**
- **B. App UX.** Radio toggle RT/DA (simple) vs show both forecasts together.
  **Recommend: toggle.**
- **C. Retrain job.** Two scheduled functions in one file vs one env-param'd
  job. **Recommend: two functions, one file.**
- **D. DA hyperparameters.** Reuse RT `TIDE_PARAMS` now; DA-specific Optuna
  re-tune is optional follow-up. **Recommend: reuse first, tune later.**

## Out of scope (for now)

- Modeling the DA→RT basis (a separate experiment).
- Using DA LMP as a *covariate* in the RT model (the earlier idea; independent
  of standing up the DA target).
- Backfilling DA history (none exists pre-launch; ~3 months is the ceiling).
