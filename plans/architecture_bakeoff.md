# Architecture bake-off — TimeXer / PatchTST / foundation models

Exploratory plan to test whether a different model architecture beats the
tuned TiDE ensemble on the West / Integrated Marketplace nodal price forecast.
Extracted from the (completed) volatility-improvements work, which established
the current champion and the evaluation harness; this is the remaining
*optional* upside — general accuracy, not tail-specific (the tail ceiling is a
data limit, proven separately).

## Status

Not started. Prerequisites are all in place from the prior work:
- **Stack:** Darts 0.45 on the GB10 (Blackwell) box with CUDA torch. Darts 0.45
  exposes `NeuralForecastModel` (wraps PatchTST / TimeXer / NBEATSx) and the
  foundation models `Chronos2Model`, `TimesFM2p5Model`, `TiRexModel`,
  `PatchTSTFMModel`.
- **Harness:** `src/evaluation.py::backtest_report` — rolling-origin West
  holdout reporting CRPS, coverage/width, MAE/RMSE/bias, and tail behavior.
  Every candidate below is scored on it, the same way, for a fair comparison.
- **Metric:** CRPS is primary (proper score), read alongside coverage.

## Baseline to beat (the current champion)

The tuned TiDE ensemble with the wider `parameters.QUANTILES` set, scored on the
full 21-day harness holdout:

| CRPS | 90% coverage | 90% width | MAE | bias | tail cov (\|x\|>$100) |
|---|---|---|---|---|---|
| ~17.3 | 0.88 | ~$103 | ~20.5 | ~0.0 | 0.23 |

A candidate has to beat CRPS ~17 (and not regress coverage) to be worth
promoting. Note the deep negatives/spikes: per-node LMP std ~$72, negatives to
about −$320, scarcity spikes over $1,400 — and only ~3 months of IM history.

## Experiments

### 1. TimeXer / PatchTST via `NeuralForecastModel`
Test both against the tuned-TiDE baseline. **TimeXer** first — it is designed
to exploit exogenous covariates (MTLF, wind/solar, load-net-RE), which drive
the duck-curve volatility. PatchTST second for the long 120-hour horizon.
**Hypothesis:** attention over exogenous drivers captures the negative-midday /
evening-ramp structure better than TiDE. **Effort:** medium-high (new model
integration + its own tuning). Reuse the single-objective CRPS Optuna study
pattern (`notebooks/model_training/model.py`) for tuning, scored on the harness.

### 2. Foundation models — zero-shot then fine-tuned
Given only ~3 months of IM data, a pretrained model's priors may beat a
from-scratch net. Test **Chronos2 / TiRex / TimesFM zero-shot first** (no
training — cheap to try as a baseline), then `enable_finetuning` on the IM
data. **Hypothesis:** competitive or better with far less data sensitivity;
possibly the best interim model until a full year of IM history exists.
**Effort:** low to try zero-shot; medium to fine-tune. VRAM is not a constraint
on the GB10 (large unified memory) — the 120–260M-param models fit comfortably.
Caveat: several foundation models are univariate or have limited covariate
support — check whether they can use the MTLF/renewable future covariates that
drive the volatility, since that is TiDE's main advantage here.

## Suggested sequencing

1. **Zero-shot foundation models first** — cheapest signal (no training). If a
   zero-shot model is already near CRPS ~17, that reframes everything.
2. **TimeXer** — the most likely architecture win given our strong exogenous
   covariates.
3. **PatchTST** and **foundation-model fine-tuning** if 1–2 are promising.
4. Promote only a candidate that beats CRPS on the harness without regressing
   coverage, via the `model_retrain.py` → `r2_promote_champion.py` flow.

## Notes / risks

- Keep the champion swap decoupled: score everything on the harness first;
  only touch `champion.json` for a clear, verified CRPS win.
- New model classes must load through the serving path
  (`load_ensemble_from_dir` matches checkpoints by filename substring
  `tide_`/`tsmixer`/`tft`) — a new architecture needs its class added to
  `MODEL_CLASS_MAP` and a matching filename, or the loader will reject it.
- Short IM history (~3 months) limits from-scratch deep nets — the main
  argument for the foundation-model route.
