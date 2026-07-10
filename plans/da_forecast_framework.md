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

## Phase 4 — Serving (app) — ✅ DONE (code; deploy pending)

- **RT champion migrated** into `models/rt/`: `models/champion.json` +
  `models/retrains/*` copied to `models/rt/*` (champion pointer rewritten;
  103 objects). Additive copy — the flat originals stay so the live app is
  unaffected until the new app deploys; clean up the flat copies post-cutover.
- `app.py`: a **Market** selector (`input.target`, Day-ahead default per
  `parameters.DEFAULT_TARGET`) at the top of the sidebar. `_do_load_data` /
  `_do_load_models` take a target; `_load_startup` depends on `input.target()`
  and reloads both the data and champion when the market changes (tracked via a
  new `loaded_target_val`); `_clear_stale_forecast` clears on switch; the
  forecast header names the market. Reuses the node dropdown + `plotting.py`.
- **e2e:** added a market-toggle test (switch DA→RT, wait for the model
  timestamp to change = reload done, forecast renders with "Real-time"). Full
  e2e suite green.
- **Deploy (pending, user-coordinated):** merge to `main` + push → Posit Connect
  redeploys the app (serves DA default + toggle). Then clean up the flat
  `models/champion.json` + `models/retrains/` once RT is confirmed on `models/rt/`.
  The Modal retrain job redeploy (both targets) is Phase 5.

## Phase 5 — Modal retrain job

## Phase 5 — Modal retrain job — ✅ DONE (code; deploy pending)

- `modal_jobs/model_retrain.py` now has two scheduled functions in the one app
  (`spp-weis-model-retrain`), sharing a `_COMMON` config: `retrain_da_weekly`
  (Sun 20:00 UTC, `env TARGET=da`, primary) and `retrain_rt_weekly` (Sun 22:00
  UTC, `env TARGET=rt`). Both run the same notebook, which selects the target
  from `TARGET`. Redeploying replaces the old single `model_retrain_weekly`
  (which wrote the flat `models/champion.json`), so future retrains land in
  `models/da/` and `models/rt/`. Collection is unchanged — DA is already gathered
  into `im/da_lmp.parquet`.
- **Deploy (pending, part of cutover):** `modal deploy modal_jobs/model_retrain.py`
  together with the Posit app redeploy, so the app (reading `models/rt|da/`) and
  the retrain (writing `models/rt|da/`) cut over together.

## Phase 6 — Tests + docs

- Unit tests for the target config + DA `create_database` branch.
- Update `src/README.md`, `CLAUDE.md` (note the `target` dimension in the model
  config + champion layout), and `.env`/deploy notes if a `TARGET` env is added.

## Phase 7 — Accuracy metrics in artifacts → champion/challenger

Persist each retrain's backtest accuracy next to its checkpoints so promotion
can become a metric-gated champion/challenger decision instead of the current
blunt `PROMOTE_CHAMPION` (first-wins / manual) flow.

**Step A — persist metrics at train time — ✅ DONE.**
`backtest_report` now returns `(per_node, aggregate, eval_meta)`; `eval_meta`
records the exact scored window (`test_start`/`test_end` — the realized-hour
range across nodes) and the eval config (`holdout_days, stride,
forecast_horizon, num_samples, interval, tail_threshold`). The retrain notebook's
scoring cell captures the return and writes **`metrics.json`** into
`models/<target>/retrains/<ts>/`, alongside `training_config.json` (kept
separate: config = provenance/inputs, metrics = evaluation results). Structure:
`{target, train_timestamp, primary_metric: 'crps', metrics: {...}, eval: {...}}`.
The existing DA champion was backfilled with a re-scored `metrics.json`.

Still optional: add `metrics.json` to the `get_loaded_models` download filter if
the app should surface the champion's CRPS.

**The test-window problem (why the date range matters).** `backtest_report`'s
holdout is defined *relative to each series' `end_time()`* (the last
`holdout_days`). Every retrain adds days, so the window slides forward — two
models trained on different dates are scored on **different** date ranges and
their stored CRPS are not directly comparable. Options, to decide before Step B:
1. **Pin an absolute test window** — a fixed held-out date range (e.g.
   `2026-06-10 .. 2026-07-01`) passed to `backtest_report` (add `test_start`/
   `test_end` params) so every model, champion or challenger, is scored on the
   identical set. Simplest path to apples-to-apples; the cost is the pinned
   window ages and must be advanced deliberately (a versioned "eval window").
2. **Record the range + re-score on demand** — keep the rolling holdout but
   record `test_start`/`test_end`, and at promote time re-score the champion on
   the challenger's exact window rather than trusting stored numbers.
Recommend **(1) a pinned, versioned test window** for the champion/challenger
comparison, with the rolling backtest kept for quick per-retrain telemetry.

**Step B — the champion/challenger framework (future).**
On retrain, treat the new model as a *challenger*:
1. Score the challenger on the current holdout.
2. **Re-score the current champion on the *same* holdout** — stored metrics from
   different dates are NOT directly comparable, because the rolling training
   window (and thus the holdout) grows over time. The only fair comparison is
   champion vs challenger evaluated together on identical data/config.
3. Apply a promotion rule: promote iff the challenger beats the champion on the
   primary metric (CRPS) by a margin, subject to guardrails (coverage within a
   tolerance of nominal; no bias/RMSE regression beyond a threshold). Otherwise
   keep the champion and log the challenger for review.
4. Replace the boolean `PROMOTE_CHAMPION` with this metric-gated promote, per
   target (`r2_promote_champion.py` can grow a `--if-better` mode).
5. Telemetry: append each retrain's metrics to a per-target
   `models/<target>/metrics_history.jsonl` to track drift over time.

**Comparability guardrails to honor:**
- Never compare across targets (DA vs RT are different scales/series);
  champion/challenger is always within one target's `models/<target>/` namespace.
- Prefer re-scoring both models together at promote time over trusting stored
  numbers computed on different windows.

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
