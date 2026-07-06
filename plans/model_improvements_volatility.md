# Model improvements for RTO West volatility — test plan

Research-backed plan for improving the West nodal price model on the new,
much more volatile RTO West / Integrated Marketplace data. **Report-only;
nothing here is implemented yet.**

## Problem statement

The RTO West market is far spikier than the retired WEIS market: per-node
LMP std ~$72 (vs WEIS ~$30), with deep solar-oversupply negatives to
about −$320 and scarcity spikes over $1,400. Two concrete weaknesses in
the current champion:

1. **Under-dispersed point forecasts.** Fixed by training IM-only (forecast
   std went from ~$6 to ~$25 vs actual ~$28), but there is likely more to
   recover with better architectures / re-tuned params.
2. **Miscalibrated intervals.** CI coverage was ~0.42 (should be ~0.90) —
   the quantile intervals are far too narrow for the tails.

## Current setup (baseline)

- **Darts 0.41.0.** TiDE ensemble (`USE_TIDE=True`, `TOP_N=5`); TSMixer/TFT
  available but off. Config in `src/parameters.py`, build in `src/modeling.py`.
- Already using: **`QuantileRegression` likelihood**, **reversible instance
  norm** (`use_reversible_instance_norm=True`), 500-sample probabilistic
  prediction, RMSE as the training `torch_metric`.
- `TIDE_PARAMS` were **Optuna-tuned on WEIS data** (lr ~1e-5, dropout
  0.36–0.47) — not re-tuned for the RTO West distribution.
- Training window is now **IM-only** (clamped to the 2026-04-01 launch,
  ~3 months / ~2,200 rows per node), which is short.

## Prerequisites — running on a fresh machine

Everything below assumes the repo runs and can reach R2. On a new box:

1. **Environment.** Python 3.11 + [uv](https://docs.astral.sh/uv/):
   `git clone`, then `uv sync` — this installs the pinned baseline
   (`darts==0.41.0`, `torch<2.6`, `optuna-integration[pytorch-lightning]`,
   `marimo`). A CUDA GPU is effectively required: the study is 100 trials,
   each fitting a TiDE ensemble. Darts/Lightning **auto-select** the GPU —
   `src/modeling.py` sets no accelerator flag, it just uses CUDA if `torch`
   sees it, so all you need is a CUDA-enabled `torch` install.
2. **Credentials.** Copy `.env.example` → `.env` and fill the R2 keys:
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT_URL` (the R2
   S3 endpoint), `AWS_DEFAULT_REGION=auto`, `AWS_S3_BUCKET=spp-weis-forecast`,
   `AWS_S3_FOLDER=""`. Without these, `de.create_database()` can't read the
   `data_im/` prefix and nothing runs. **No local data collection is needed**
   — the scheduled Modal jobs keep R2 current; the notebook reads whatever is
   in `data_im/` at run time (so the backtest window tracks R2's freshness).
3. **Run the Optuna study.** The notebook is
   `notebooks/model_training/model.py`. Run it with
   `uv run marimo edit notebooks/model_training/model.py` (interactive) or
   `uv run marimo run …` (headless). Its knobs are **constants in the first
   cell, not env vars**: `MODEL_TYPE="tide"`, `NUM_TRIALS=100`,
   `RUN_EXP=True`, `REMOVE_PRIOR_MODELS=True`. Local outputs (gitignored,
   live only on that machine): `spp_trials.db` (the Optuna sqlite study),
   `study_csv/` (per-trial metrics), `optuna/<type>/` (saved trial models).
   The two objectives are `MAE` and `CI_ERROR` (coverage error from
   `get_ci_err`, `src/modeling.py:483`) — CRPS is **not** available on this
   pinned Darts (needs the 0.45 upgrade, Experiment 0).
4. **Land the winning params.** `TIDE_PARAMS` in `src/parameters.py` is a
   **list of dicts, one per ensemble member** (`TOP_N=5`). Copy the top
   trials' `params` in by hand — there is no auto-writeback from the study.
5. **Reproduce before changing anything.** Retrain once on the current
   pinned stack (`notebooks/model_training/model_retrain.py`) and confirm it
   loads/serves before starting the Darts upgrade. Keep `requirements.txt` +
   `manifest.json` synced with `pyproject.toml` if any runtime dep changes
   (Posit Connect deploy).

## What's new in Darts since 0.41 (latest 0.45.0, 2026-06-19)

Relevant additions, with the version that introduced them:

| Feature | Ver | Why it matters here |
|---|---|---|
| **`NeuralForecastModel`** — wraps NBEATSx, **PatchTST**, **TimeXer**, KAN | 0.42 | SOTA transformers with covariate + probabilistic support. **TimeXer** is purpose-built for *exogenous* variables (we have MTLF/MTRF/renewables), PatchTST is strong on long, volatile horizons. |
| **Foundation models**: `Chronos2Model` (0.39), `TimesFM2p5Model` (0.41), `TiRexModel` (0.44), `PatchTSTFMModel` (0.45) | 0.39–0.45 | Pre-trained, **zero-shot** (no training) or fine-tunable (`enable_finetuning`, 0.42). Strong priors help exactly our weak spot: only ~3 months of IM data. |
| **Conformal prediction**: `ConformalQRModel`, `ConformalNaiveModel` | 0.32 | Wraps any model to produce **calibrated** prediction intervals — a direct fix for the 0.42 coverage. |
| **CRPS / MCRPS** probabilistic metrics | 0.44 | Proper scoring of the *whole* predictive distribution — the right metric for volatility/CI quality, better than RMSE alone. |
| Dict-based **RINorm** hyperparameters | 0.42 | Finer control of instance normalization for non-stationary prices. |
| Multi-quantile for CatBoost/XGB | 0.44 | Cheap gradient-boosted quantile baselines. |
| Classification models (Sklearn/CatBoost/LightGBM/XGB) | 0.37 | Could flag negative-price / spike *regimes* as a covariate or a two-stage model. |

Sources: [Darts release notes](https://unit8co.github.io/darts/release_notes/RELEASE_NOTES.html),
[Darts changelog](https://github.com/unit8co/darts/blob/master/CHANGELOG.md),
[Darts docs](https://unit8co.github.io/darts/).

## Evaluation harness (build first, shared by every experiment)

Without a fair comparison we can't tell wins from noise. Before testing
models, stand up a West holdout backtest that reports, per node and
aggregated:

- **CRPS / MCRPS** (probabilistic accuracy — primary metric),
- **CI coverage** at the 90% interval (target ~0.90; today ~0.42) and
  interval width,
- **MAE / RMSE / bias** of the median (point accuracy),
- **tail behavior**: error conditioned on |actual| > some threshold and on
  negative-price hours specifically.

Backtest on a rolling origin over the IM period (e.g. last 3–4 weeks) so
every candidate is scored on the same volatile windows.

**Ordering caveat:** `CRPS`/`MCRPS` are Darts **0.44+** metrics, so the full
harness can only be built *after* Experiment 0 (the 0.45 upgrade). If you
want to start the IM Optuna re-tune (Experiment 2) on today's pinned
`darts==0.41.0` first, use the metrics that already exist — `MAE` +
`get_ci_err` coverage (the notebook's current two objectives) — and add
CRPS/tail scoring to the harness once the upgrade lands. Don't block the
re-tune on the upgrade; just don't expect CRPS numbers before it.

## Experiments, prioritized (value / effort)

### 0. Upgrade Darts 0.41.0 → 0.45.0 — **do this first** (prerequisite for everything)
Unlocks conformal, NeuralForecast, foundation models, and CRPS, and has to
happen regardless, so it is the first task — the rest of the plan assumes the
0.45 stack. It is also the highest-risk step because of how the app loads
models, so sequence it carefully rather than merging a bare version bump.

**The serving load path (why the bump is load-bearing).** The Shiny app loads
its model on startup via
`app.py::_do_load_models` → `utils.download_champion_checkpoints` (reads
`S3_models/champion.json`, pulls the `champion_artifact_folder` checkpoints
from R2) → `modeling.load_ensemble_from_dir` → `model_class.load(...,
map_location="cpu")` for each `.pt`, combined into a `NaiveEnsembleModel`.
Three specifics make this fragile across a Darts upgrade:

1. **Checkpoint compatibility.** The live champion's checkpoints were written
   by Darts **0.41** (Darts saves a `.pt` + a companion `.pt.ckpt` Lightning
   checkpoint — both are downloaded and must stay together). Darts/Lightning
   model `.load()` is **not guaranteed to read checkpoints across minor
   versions** (pickled class signatures, RIN/`QuantileRegression` kwargs, and
   Lightning's checkpoint schema can all shift). Assume 0.45 may **fail or
   silently mis-load** the 0.41 champion. The fix is not to make old
   checkpoints load — it is to **retrain a fresh champion under 0.45** and
   promote that, never to serve 0.41 artifacts from a 0.45 runtime.
2. **Posit auto-deploys on merge to `main`.** The app redeploys from `main`,
   so **merging the dependency bump alone would upgrade the live app to 0.45
   while champion.json still points at 0.41 checkpoints** → the app breaks on
   the next startup `.load()`. This is the failure to avoid.
3. **Silent ensemble-member drop.** `load_ensemble_from_dir` matches each
   checkpoint to a class by filename substring (`tide_`, `tsmixer`, `tft`)
   with **no else branch** — a file that doesn't match is skipped silently, so
   a renamed/re-serialized checkpoint yields a smaller ensemble with no error.
   After the upgrade, assert the rebuilt ensemble has the expected member
   count (`len(ens.forecasting_models) == TOP_N`) before trusting it.

**Safe upgrade sequence (mirrors the RTO West `PROMOTE_CHAMPION=false`
staging):**
1. Branch; bump `darts==0.45.0` in `pyproject.toml` (and keep
   `requirements.txt` + `manifest.json` in sync for the Posit deploy). Resolve
   the `torch`/`lightning` pins 0.45 requires.
2. Retrain on the 0.45 stack to a **staged** timestamped folder with
   `PROMOTE_CHAMPION=false` (does not touch the live champion.json).
3. Verify the new serving code loads the staged model end to end: run the app
   against the staged folder, confirm `load_ensemble_from_dir` returns all
   `TOP_N` members and predicts, and sanity-check a forecast.
4. **Promote and deploy together** — merge the bump to `main` *and* repoint
   champion.json at the staged 0.45 folder
   (`scripts/r2_promote_champion.py <ts> --promote`) as one coordinated step,
   so the 0.45 app and 0.45 champion go live simultaneously. Keep the prior
   0.41 folder for one-command revert if the deploy regresses.

Only after the app reproduces on 0.45 do the harness/experiments below build
on top.

### 1. Conformal intervals — `ConformalQRModel` (highest value / lowest effort)
Wrap the existing TiDE ensemble output in conformal quantile regression to
fix the interval calibration. **Hypothesis:** coverage 0.42 → ~0.90 with
honest widths, no change to the point model. **Metric:** coverage, width,
CRPS. This is the cheapest, most direct win for the CI problem.

**Works on the current ensemble, and does not need the Darts upgrade.**
`ConformalQRModel` accepts any pre-trained `GlobalForecastingModel` and has no
ensemble exclusion; the served model is a native `NaiveEnsembleModel` built
predict-ready in `load_ensemble_from_dir`, so it qualifies directly. The one
hard requirement — `ConformalQRModel` needs `model.supports_probabilistic_
prediction` — is met because an `EnsembleModel` reports that `True` only when
*all* sub-models are probabilistic, and every TiDE member uses
`QuantileRegression`. Conformal shipped in Darts **0.32**, so it is already
importable on the pinned **0.41.0** — prototype it now; only CRPS scoring
waits on the 0.45 upgrade. Implementation notes: assert
`ens.supports_probabilistic_prediction` and `ens._fit_called` before wrapping;
carve the conformal calibration series from the West holdout (not the training
window); keep `num_samples` high (the champion uses 500) so the ensemble's
averaged-sample quantiles are stable.

### 2. Re-tune TiDE on IM data (Optuna) — the already-deferred item
Current params are WEIS-tuned. Re-run the Optuna study on IM-only data,
tuning `lr`, `dropout`, `hidden_size`, the temporal widths, **and** the
RINorm dict and the quantile set (wider tails for the −$320 dips).
**Hypothesis:** meaningful CRPS/MAE gain from params matched to the real
distribution. **Effort:** medium (GPU hours), but reuses the existing model.

**Notebook is `notebooks/model_training/model.py` (the Optuna study).** It
already flows through the current `de.create_database` / `prep_lmp` /
`prep_all_df` / `get_train_test_all` functions, so it inherits the IM-only
West clamp and `MODEL_APP_NODES` scope automatically, and its objective is
already multi-objective (`directions=["minimize","minimize"]`, targets
`MAE` + `CI_ERROR`) — matching the two weaknesses above. It could run as-is,
but before a serious IM re-tune make three edits:

1. **Re-test outlier clipping** — `de.prep_all_df(con, clip_outliers=True)`
   (line ~218) and `de.get_train_test_all(con, clip_outliers=True)`
   (line ~256) clip LMP to the 0.25% / 99.75% quantiles. **This was a
   deliberate, tested choice on WEIS data: trimming the tails improved point
   accuracy without hurting CI coverage.** The concern is that it was
   validated on the *WEIS* distribution — on the far spikier IM data those
   same tails (−$320 dips, >$1,400 spikes) are the signal we now want to
   capture, and clipping also de-tails the series the tuner is *scored* on.
   So treat this as a hypothesis to re-run, not an obvious flip: tune once
   with `clip_outliers=True` and once with `False` (or a much wider clip) and
   compare on the harness — MAE/CRPS **and** CI coverage/tail error. Keep
   clipping only if it still wins on IM. **This is the load-bearing
   experiment.**
2. **Widen the search space** in `objective_tide` — the current bounds
   (`lr 1e-5…5e-5`, `dropout 0.35…0.5`, `n_epochs 6…20`) are WEIS-era ranges
   from a less volatile, multi-year series. Widen them (especially
   `n_epochs`, given only ~3 months of IM data) and extend the tuned quantile
   set toward wider tails.
3. **Rename the stale `spp_weis` identifiers** (cosmetic) — the study name
   `f"spp_weis_{MODEL_TYPE}"` (lines ~670, ~698) should become `spp_west` so
   new IM trials don't share a `spp_trials.db` study with the old WEIS runs;
   and delete the dead `MODEL_NAME = "spp_weis"` (line ~42), which is never
   returned from its cell and never used.

Best params are logged + written to `study_csv/`; the handoff into
`TIDE_PARAMS` in `src/parameters.py` stays manual (copy the winning trial's
params). No plumbing change needed there.

### 3. TimeXer / PatchTST via `NeuralForecastModel`
Test both against the tuned-TiDE baseline. **TimeXer** first — it is
designed to exploit exogenous covariates (MTLF, wind/solar, load-net-RE),
which drive the duck-curve volatility. PatchTST second for the long
120-hour horizon. **Hypothesis:** attention over exogenous drivers captures
the negative-midday / evening-ramp structure better than TiDE.
**Effort:** medium-high (new model integration + its own tuning).

### 4. Foundation models — zero-shot then fine-tuned
Given only ~3 months of IM data, a pretrained model's priors may beat a
from-scratch net. Test **Chronos2 / TiRex / TimesFM zero-shot first**
(no training — cheap to try as a baseline), then `enable_finetuning` on the
IM data. **Hypothesis:** competitive or better with far less data
sensitivity; possibly the best interim model until a full year of IM
history exists. **Effort:** low to try zero-shot; medium to fine-tune; watch
model size/VRAM on the RTX 3080 (some are 120–260M params).

### 5. Regime-aware modeling (research spike, lower priority)
The deep negatives are a distinct regime. Options to explore: a
classification model (0.37) predicting negative/spike hours as an extra
future covariate, or an asymmetric/heavy-tailed quantile set. **Hypothesis:**
explicitly modeling the negative-price regime reduces tail error.
**Effort:** high, exploratory.

## Suggested sequencing

1. **Upgrade Darts to 0.45 first (Experiment 0).** Reproduce the current
   pipeline on the pinned stack, then bump on a branch and follow the staged
   `PROMOTE_CHAMPION=false` → verify-load → **promote-and-deploy-together**
   sequence in Experiment 0 — never merge the bare version bump, or the live
   app upgrades to 0.45 while champion.json still points at 0.41 checkpoints.
   This unlocks CRPS, conformal, and the new models, so everything else builds
   on it.
2. Build the evaluation harness (CRPS + coverage + tail metrics) on a West
   holdout — reused by everything below. (The IM Optuna re-tune can start in
   parallel on 0.41 using `MAE` + `get_ci_err`; see the ordering caveat above.)
3. Quick wins in parallel: **conformal intervals** (fixes coverage) and the
   **IM Optuna re-tune** (fixes point accuracy) — both build on today's TiDE.
4. Architecture bake-off: **TimeXer / PatchTST**, and **zero-shot foundation
   models** as a strong baseline, scored on the same harness.
5. Promote whatever wins CRPS + coverage on the holdout, using the
   `model_retrain.py` flow: `PROMOTE_CHAMPION=false` stages the checkpoints in
   a timestamped `model_retrains/<ts>/` folder without touching the live
   model; setting it true (the default) writes `S3_models/champion.json`
   pointing at that folder. To promote a staged model or **revert** to a prior
   one without retraining, run
   `python scripts/r2_promote_champion.py <timestamp> --promote`
   (`--list` shows the available folders, `--show` prints the current
   champion; it dry-runs by default and validates the target folder is
   non-empty before repointing).

## Risks / notes

- **Darts upgrade** is load-bearing — see Experiment 0 for the full serving
  path and the staged promote-and-deploy sequence. Key traps: the live app
  loads via `app.py::_do_load_models` → `download_champion_checkpoints` →
  `load_ensemble_from_dir` (**not** the mlflow `DartsGlobalModel` wrapper,
  which is unused on the serving path); 0.45 may not read 0.41 checkpoints, so
  retrain fresh rather than porting old artifacts; and `load_ensemble_from_dir`
  silently drops any checkpoint whose filename misses the class substrings, so
  assert the ensemble has `TOP_N` members after loading.
- **Short IM history (~3 months)** limits from-scratch deep nets — this is the
  strongest argument for foundation models and for keeping the ensemble.
- **VRAM**: foundation models (120–260M params) vs the 10 GB RTX 3080 — may
  need CPU inference or the smaller variants.
- Keep the quick, safe wins (conformal + re-tune) decoupled from the riskier
  architecture swaps so calibration can ship even if the bake-off stalls.
