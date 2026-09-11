---
name: tune-parameters
description: Run a hyperparameter sweep for a forecast target (da/rt) and, if it wins, promote a re-tuned champion. Orchestrates the Optuna study, bakes the top-N params into parameters.py, retrains, scores vs the current champion on the same harness, and promotes only if better.
disable-model-invocation: false
allowed-tools: Read, Edit, Bash
argument-hint: [da|rt] [--trials N] [--objective MODE]
---

# Tune a forecast target

End-to-end hyperparameter tuning for one forecast target (`parameters.TARGETS`:
`da` day-ahead — default/primary — or `rt` real-time). The deterministic param
edit is done by `scripts/tune_parameters.py`; this skill orchestrates the flow
around it and makes the promote decision.

**Target** = the argument (`da` if omitted). **Trials** = `--trials N` (default
100). **Objective** = `--objective MODE` (default `mae_ci`), a key of
`selection.OBJECTIVES` — how trials are ranked. Everything runs on the local GPU
box.

The objective mode is part of the study name, so switching modes starts a
separate study rather than mixing incomparable trials. Modes:

| Mode | Ranks on |
|------|----------|
| `mae_ci` (default) | MAE + 0.25x coverage error at the 80% and 90% bands |
| `crps_ci` | CRPS + the same calibration term |
| `crps` | CRPS alone (single-objective) — kept runnable as a baseline |

## Steps

1. **Run the Optuna study.** Launch the TiDE study for the target in the
   background (it is long — ~1 min/trial, so ~1.5 h for 100):

   ```
   TARGET=<t> OBJECTIVE_MODE=<mode> uv run python -c "import sys; sys.path[:0]=['.','src']; from notebooks.model_training.model import app; app.run()"
   ```

   The study writes to `sqlite:///spp_trials.db`, study name
   `spp_west[_da]_tide_<mode>`. Monitor progress via the **DB, not the log** (the
   log is progress-bar noise and its Optuna timestamps are **local time** — do
   not compare them to `date -u`). `study.best_value` **raises** under the
   default two-objective modes, so rank the Pareto front by the composite:

   ```
   uv run python -c "import sys, optuna; sys.path.insert(0,'src'); import selection; \
   m=selection.resolve_mode('<mode>'); s=optuna.load_study(study_name='<name>', storage='sqlite:///spp_trials.db'); \
   d=[t for t in s.trials if str(t.state)=='TrialState.COMPLETE']; \
   print(len(d),'complete, best score', round(min(selection.selection_score(t.values, m) for t in d),3))"
   ```

   Wait for it to reach the trial count (or plateau).

2. **Bake the winners** into `parameters.py` (dry-run first, then write):

   ```
   uv run python scripts/tune_parameters.py --target <t> --objective <mode>
   uv run python scripts/tune_parameters.py --target <t> --objective <mode> --write
   ```

   `--objective` picks both the study to read and the formula the trials are
   ranked by, so a bake can never rank one mode's trials by another's.

   Then confirm the repo is still healthy: `uv run ruff check src/parameters.py`
   and `uv run pytest tests/unit -q`.

3. **Retrain — it self-gates the promotion.** Run the retrain with
   `PROMOTE_CHAMPION=true`:

   ```
   TARGET=<t> OBJECTIVE_MODE=<mode> PROMOTE_CHAMPION=true uv run python <retrain runner>
   ```

   (Runner: a 3-line script that sets the env, `load_dotenv`, then
   `from notebooks.model_training.model_retrain import app; app.run()`.) The
   retrain trains the candidate, uploads it to `models/<t>/retrains/<ts>/` with a
   `metrics.json`, then runs `evaluation.compare_candidate_to_champion` — a fast
   backtest of the candidate **and** the current champion on the same recent
   window over the fixed `node_list.EVAL_NODES` — and **only promotes if the
   candidate wins on the mode's composite score** (the first champion for a
   target promotes unconditionally). Watch the log for the
   `Promote gate (<t>, <mode>): candidate score … vs champion … -> PROMOTE /
   KEEP champion` line. Use the **same** mode the study ran under — the gate
   must judge by what the sweep optimized.

4. **Report** the top trials and the gate's promote decision. If params changed,
   remind the user to commit `src/parameters.py` and (at cutover) redeploy so the
   scheduled retrain uses the tuned params. To override the gate — promote a
   staged model by hand, or revert — use
   `python scripts/r2_promote_champion.py <ts> --target <t> --promote`.

## Notes

- `tune_parameters.py` only edits the marked `# >>> TIDE_PARAMS_<TARGET> >>>`
  block; it never touches the other target's params.
- Never compare scores across targets (DA and RT are different scales) or
  across objective modes (different formulas).
- The study is resumable **within one mode**: to add trials, set
  `REMOVE_PRIOR_MODELS=False` in `notebooks/model_training/model.py` and re-run
  with the same `OBJECTIVE_MODE` (Optuna continues via the sqlite storage).
  `REMOVE_PRIOR_MODELS=True` starts fresh.
- **Never resume a study under a different mode.** Optuna *silently ignores* a
  changed `directions` on an existing study: it keeps the stored objective
  count, every trial then fails with "number of the values … did not match the
  number of the objectives", and `study.optimize` does not propagate that — the
  sweep would burn ~1.5 h and exit cleanly with nothing usable. The mode-suffixed
  study name prevents this, and a guard in the notebook catches a hand-edited
  name; do not work around either.
- Every trial records MAE, CRPS and coverage error at all four diagnostic bands
  (50/80/90/95), whichever mode ran. So a finished study can be **re-ranked**
  under a different weighting offline — no re-running needed to compare
  objectives.
