# Model improvements for RTO West volatility — test plan

Research-backed plan for improving the West nodal price model on the new,
much more volatile RTO West / Integrated Marketplace data.

## Status (updated 2026-07-06)

Work is underway on branch `rto-west-volatility-improvements`.

- **Experiment 0 (Darts 0.41→0.45 upgrade) — DONE locally, not yet
  deployed.** Upgraded to `darts==0.45.0` on a Blackwell **GB10** box
  (ARM/aarch64) with a CUDA-enabled `torch==2.11.0+cu128` from PyTorch's
  cu128 index (the default PyPI aarch64 torch is CPU-only and cannot see
  the GB10). Loader hardened + fixed for torch 2.6+ (`weights_only`); a
  fresh 0.45 baseline champion was retrained (4.36 min, 5×TiDE) and
  verified to load+predict via the serving path. See the Experiment 0
  section for the full findings and what changed vs. the original plan.
- **Evaluation harness — DONE.** `src/evaluation.py::backtest_report` (rolling
  West holdout: CRPS, coverage/width, MAE/RMSE/bias, tail). Baseline scored on
  the staged 0.45 champion: **CRPS 61.4**, 90% coverage ~1.00 / width ~$1,412
  — **over-dispersed** (inverts the plan's original "intervals too narrow").
- **Experiment 2 groundwork — DONE.** `CLIP_OUTLIERS` toggle, widened
  Optuna search space, and single-source-of-truth study name landed in
  `notebooks/model_training/model.py`.
- **Deferred:** Posit deploy pins (`requirements.txt` / `manifest.json`)
  are NOT yet regenerated — they target the CPU deploy host, not the local
  cu128/aarch64 wheels, and belong to the coordinated promote-and-deploy
  step. The live app still runs darts 0.41; do not merge to `main` until
  the deploy pins are regenerated and a 0.45 champion is promoted together.

Everything below the status block is the original test plan, with the
Experiment 0 section rewritten to record the outcome.

## Problem statement

The RTO West market is far spikier than the retired WEIS market: per-node
LMP std ~$72 (vs WEIS ~$30), with deep solar-oversupply negatives to
about −$320 and scarcity spikes over $1,400. Two concrete weaknesses in
the current champion:

1. **Under-dispersed point forecasts.** Fixed by training IM-only (forecast
   std went from ~$6 to ~$25 vs actual ~$28), but there is likely more to
   recover with better architectures / re-tuned params.
2. **Miscalibrated intervals.** The *prior* champion under-covered (CI
   coverage ~0.42 at the 80% interval). **Update (harness baseline, 2026-07-06):
   the fresh IM-only 0.45 champion over-corrects — it now over-covers** (90%
   band ≈ 1.00 coverage at ~$1,412 wide). Either way the intervals are
   miscalibrated; the current job is to *sharpen* them. See the evaluation
   harness section for the measured baseline.

## Current setup (baseline)

- **Darts 0.45.0** (upgraded from 0.41.0 — Experiment 0). TiDE ensemble
  (`USE_TIDE=True`, `TOP_N=5`); TSMixer/TFT available but off. Config in
  `src/parameters.py`, build in `src/modeling.py`.
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
   `git clone`, then `uv sync` — this installs `darts==0.45.0`,
   `torch>=2.7` (from the cu128 index configured in `pyproject.toml`),
   `optuna-integration[pytorch-lightning]`, `marimo`. A CUDA GPU is
   effectively required: the study is 100 trials, each fitting a TiDE
   ensemble. Darts/Lightning **auto-select** the GPU — `src/modeling.py`
   sets no accelerator flag, it just uses CUDA if `torch` sees it.
   **Caveat (hardware-specific, learned the hard way on the GB10):** a
   plain `torch` install is often *not* CUDA-enabled. On ARM/aarch64 the
   default PyPI torch wheel is **CPU-only**, and Blackwell parts (GB10,
   sm_121) need CUDA 12.8+ — so `pyproject.toml` pins torch to PyTorch's
   **cu128** index. Verify with
   `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   before assuming the GPU is live; a CPU-only torch silently trains on CPU.
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

## Evaluation harness (build first, shared by every experiment) — ✅ DONE

Built as `src/evaluation.py::backtest_report(model, series, past_covariates,
future_covariates, ...)`: a rolling-origin backtest (daily origins over a
configurable holdout, default 21 days; `num_samples` default 200) that scores
any Darts model supporting `historical_forecasts`. It reports, per node and
aggregated, using Darts 0.45 metrics (`mcrps`, `mic`, `miw`, `mae`/`rmse`/
`merr` at `q=0.5`) plus a numpy path for the tail conditioning Darts can't do:

- **CRPS** (probabilistic accuracy — primary metric),
- **CI coverage** and **interval width** at the 90% interval (0.05–0.95),
- **MAE / RMSE / bias** of the median (point accuracy),
- **tail behavior**: median error and coverage conditioned on
  `|actual| > tail_threshold` (default $100) and on negative-price hours.

Unit-tested (`tests/unit/test_evaluation.py`) and wired into
`model_retrain.py` as a non-blocking post-promote cell, so every retrain logs
its holdout metrics. Experiments call `backtest_report` directly from the
study notebook — there is deliberately **no standalone eval script** (it would
be a third copy of the data-loading glue the notebooks already have).

**Baseline — staged 0.45 champion `model_retrains/2026-07-06_14-34-15/`**
(all 10 nodes, 21-day holdout, 200 samples; the number every experiment must
beat):

| Metric | Value |
|---|---|
| CRPS (primary) | **61.4** |
| 90% coverage | ~1.00 |
| 90% interval width | ~$1,412 |
| MAE / RMSE (median) | 33.3 / 69.7 |
| Bias | ~−0.3 |
| Tail MAE (`|x|`>$100) / coverage | 311 / 0.77 |
| Neg-hour MAE / coverage | 38 / ~1.00 |

**Key finding — the CI problem inverted.** The fresh IM-only 0.45 baseline is
**over-dispersed**, not under-dispersed: the 90% band covers ~100% of hours at
~$1,412 wide (yet still misses ~23% of the >$100 spike hours). Point bias is
essentially zero (the IM-only retrain fixed the old under-forecasting). So the
calibration work is about **sharpening / tightening** the intervals, not
widening them — conformal should *narrow* here. This updates the problem
statement below, whose "intervals too narrow (0.42 coverage)" described the
*prior* champion, measured differently (`get_ci_err` at the 80% interval).

**Ordering caveat — now resolved.** `CRPS`/`MCRPS` are Darts **0.44+**
metrics; the 0.45 upgrade (Experiment 0) is done, so `darts.metrics.crps` is
importable and the full harness (CRPS + tail scoring alongside the existing
`MAE` + `get_ci_err` coverage) can be built now.

## Experiments, prioritized (value / effort)

### 0. Upgrade Darts 0.41.0 → 0.45.0 — ✅ DONE locally (2026-07-06), deploy pending
Unlocks conformal, NeuralForecast, foundation models, and CRPS. Done on branch
`rto-west-volatility-improvements`; the live app on `main` is still 0.41 until
the coordinated promote-and-deploy step (below).

**What was actually done**
- `pyproject.toml`: `darts==0.45.0`, `torch>=2.7`, plus a `[[tool.uv.index]]`
  for PyTorch's **cu128** wheels (the GB10/Blackwell GPU needs CUDA 12.8+ and
  the default aarch64 torch is CPU-only). Resolved to `torch==2.11.0+cu128`;
  `torch.cuda.is_available()` is `True` on the GB10. `ConformalQRModel` and the
  `crps` metric both import on 0.45.
- **Loader fix + hardening** in `src/modeling.load_ensemble_from_dir` (and the
  Optuna study's trial reload in `model.py`): pass `weights_only=False`, and
  raise on a checkpoint that matches no model-class substring or on an empty
  ensemble (instead of the plan's original literal `len==TOP_N` assert, which
  would wrongly fail if TSMixer/TFT are re-enabled — member count is
  *enabled-types × TOP_N*).
- Retrained a fresh 0.45 baseline champion (5×TiDE, **4.36 min** on the GB10)
  to a staged folder `model_retrains/2026-07-06_14-34-15/` with
  `PROMOTE_CHAMPION=false`, and independently verified it **loads (5 members)
  and predicts** (120-h horizon) through the real serving path.

**Corrected root-cause (differs from the original plan's guess).** The plan
assumed 0.41 checkpoints might be *format*-incompatible with 0.45 and that the
retrain was needed to make them load. The reality is two *separate* issues:

1. **torch 2.6+ `weights_only` default (a code problem, not a data problem).**
   torch 2.6 flipped `torch.load`'s default to `weights_only=True`, which
   refuses to unpickle Darts' `QuantileRegression` likelihood in the Lightning
   checkpoint. This breaks loading **any** probabilistic Darts model — a
   *freshly-trained 0.45* one too, not just the old 0.41 champion. Retraining
   does **not** fix it; the fix is `weights_only=False` in the loader (safe
   here — the checkpoints are our own artifacts from our private R2 bucket).
2. **Stale pickled encoder (what actually mandates the retrain).** Once (1) is
   fixed, the 0.41 champion *loads* fine but **fails at predict time**: its
   `add_encoders` `Scaler` was pickled by 0.41 and lacks the `_columns`
   attribute 0.45's `Scaler.transform` now expects
   (`AttributeError: 'Scaler' object has no attribute '_columns'`). So the plan's
   conclusion — **retrain fresh under 0.45, never serve 0.41 artifacts** — holds,
   but the failure surfaces at *inference*, not at load. A fresh 0.45 champion
   (with 0.45-pickled encoders) predicts cleanly, as verified above.

**Other serving-path facts that held up:** the load path is
`app.py::_do_load_models` → `utils.download_champion_checkpoints` →
`modeling.load_ensemble_from_dir` → `model_class.load(..., map_location="cpu")`
→ `NaiveEnsembleModel`; and Posit auto-deploys on merge to `main`, so **merging
the bump alone would upgrade the live app to 0.45 while champion.json still
points at 0.41 checkpoints** → broken startup. Still the failure to avoid.

**Remaining to finish Experiment 0 (the coordinated deploy):**
1. **Regenerate the Posit deploy pins** — `requirements.txt` + `manifest.json`
   for `darts==0.45.0` against the **CPU deploy host** (x86), NOT the local
   cu128/aarch64 wheels. Do not hand-copy the local `torch==2.11.0+cu128`.
2. **Promote and deploy together** — merge to `main` *and* repoint champion.json
   at the staged 0.45 folder
   (`python scripts/r2_promote_champion.py 2026-07-06_14-34-15 --promote`) as
   one coordinated step, so the 0.45 app and 0.45 champion go live together.
   Keep the prior 0.41 folder for one-command revert.
   **Caveat to check at deploy time:** confirm the staged champion also loads on
   the deploy host's torch — a checkpoint saved by torch 2.11 must be readable
   by whatever CPU torch the Posit pins resolve to.

Only after the app reproduces on 0.45 do the harness/experiments below build
on top.

### 1. Conformal intervals — `ConformalQRModel` (highest value / lowest effort)
Wrap the existing TiDE ensemble output in conformal quantile regression to
fix the interval calibration. **Hypothesis (revised by the baseline):** the
0.45 baseline *over*-covers (90% band ≈ 1.00 at ~$1,412 wide), so conformal
should **tighten** the intervals toward honest ~0.90 coverage and much
narrower width — while improving CRPS. **Metric:** coverage, width, CRPS on
the harness. Still the cheapest, most direct win for the CI problem.

**Works on the current ensemble, and does not need the Darts upgrade.**
`ConformalQRModel` accepts any pre-trained `GlobalForecastingModel` and has no
ensemble exclusion; the served model is a native `NaiveEnsembleModel` built
predict-ready in `load_ensemble_from_dir`, so it qualifies directly. The one
hard requirement — `ConformalQRModel` needs `model.supports_probabilistic_
prediction` — is met because an `EnsembleModel` reports that `True` only when
*all* sub-models are probabilistic, and every TiDE member uses
`QuantileRegression`. Conformal (`ConformalQRModel`) and the `crps` metric
are both importable on the now-installed **0.45.0** — prototype it against the
staged 0.45 baseline champion (`model_retrains/2026-07-06_14-34-15/`).
Implementation notes: assert
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
but before a serious IM re-tune make three edits. **Edits 2 and 3 are DONE;
edit 1 (the clip A/B) is the remaining experiment to run:**

1. **Re-test outlier clipping** — ⏳ TODO (the load-bearing experiment). The
   two clip calls now read a single `CLIP_OUTLIERS` toggle in the notebook's
   first cell (added with edit 2/3), so the A/B is a one-line flip. They clip
   LMP to the 0.25% / 99.75% quantiles. **This was a
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
2. **Widen the search space** in `objective_tide` — ✅ DONE. Bounds widened
   from the WEIS-era ranges to `lr` log `1e-5…1e-3`, `n_epochs 6…60`, `dropout
   0.1…0.5`. Still TODO if desired: extend the *tuned quantile set* toward
   wider tails (the quantile list is currently fixed in `src/modeling.py`'s
   build functions, not tuned per-trial — changing it touches the served
   model's output distribution and `get_ci_err`, so validate on the harness).
3. **Rename the stale `spp_weis` identifiers** — ✅ DONE. The study name now
   derives from `parameters.MODEL_NAME` (`f"{parameters.MODEL_NAME}_{MODEL_TYPE}"`,
   single source of truth) instead of a hardcoded literal, so it is already
   `spp_west_*` and won't share a `spp_trials.db` study with old WEIS runs; the
   dead `MODEL_NAME = "spp_weis"` constant was deleted.

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
history exists. **Effort:** low to try zero-shot; medium to fine-tune. VRAM is
no longer a constraint on the current **GB10** box (large unified memory);
the 120–260M-param models fit comfortably.

### 5. Regime-aware modeling (research spike, lower priority)
The deep negatives are a distinct regime. Options to explore: a
classification model (0.37) predicting negative/spike hours as an extra
future covariate, or an asymmetric/heavy-tailed quantile set. **Hypothesis:**
explicitly modeling the negative-price regime reduces tail error.
**Effort:** high, exploratory.

## Suggested sequencing

1. **Upgrade Darts to 0.45 (Experiment 0)** — ✅ DONE locally on branch
   `rto-west-volatility-improvements`; a staged 0.45 baseline champion exists
   and loads+predicts. **Still to do: the coordinated deploy** — regenerate the
   Posit pins for the CPU host, then promote-and-deploy-together (never merge
   the bare bump, or the live app upgrades to 0.45 while champion.json still
   points at 0.41 checkpoints). See Experiment 0.
2. **Build the evaluation harness** (CRPS + coverage + tail metrics) on a West
   holdout — ✅ DONE (`src/evaluation.py`). Baseline scored: CRPS 61.4, 90%
   coverage ~1.00 / width ~$1,412 (over-dispersed), MAE 33.3, bias ~0. See the
   evaluation-harness section.
3. Quick wins in parallel: **conformal intervals** (fixes coverage) and the
   **IM Optuna re-tune** (fixes point accuracy — the re-tune prep is done; run
   the study, including the `CLIP_OUTLIERS` A/B) — both build on today's TiDE.
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
  path and the staged promote-and-deploy sequence. Key traps (updated with what
  was found): the live app loads via `app.py::_do_load_models` →
  `download_champion_checkpoints` → `load_ensemble_from_dir` (**not** the mlflow
  `DartsGlobalModel` wrapper, which is unused — dead code, a delete candidate);
  the real breakage was **torch 2.6+ `weights_only=True`** (fixed with
  `weights_only=False`) plus **0.41-pickled encoders that fail at predict time**
  (fixed by retraining fresh under 0.45 — done). `load_ensemble_from_dir` now
  **raises** on an unmatched checkpoint or empty ensemble instead of silently
  dropping members.
- **Short IM history (~3 months)** limits from-scratch deep nets — this is the
  strongest argument for foundation models and for keeping the ensemble.
- **VRAM is not a constraint** on the current GB10 (large unified memory); the
  earlier RTX-3080 concern no longer applies.
- Keep the quick, safe wins (conformal + re-tune) decoupled from the riskier
  architecture swaps so calibration can ship even if the bake-off stalls.
