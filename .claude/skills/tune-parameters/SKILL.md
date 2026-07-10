---
name: tune-parameters
description: Run a hyperparameter sweep for a forecast target (da/rt) and, if it wins, promote a re-tuned champion. Orchestrates the Optuna study, bakes the top-N params into parameters.py, retrains, scores vs the current champion on the same harness, and promotes only if better.
disable-model-invocation: false
allowed-tools: Read, Edit, Bash
argument-hint: [da|rt] [--trials N]
---

# Tune a forecast target

End-to-end hyperparameter tuning for one forecast target (`parameters.TARGETS`:
`da` day-ahead — default/primary — or `rt` real-time). The deterministic param
edit is done by `scripts/tune_parameters.py`; this skill orchestrates the flow
around it and makes the promote decision.

**Target** = the argument (`da` if omitted). **Trials** = `--trials N` (default
100). Everything runs on the local GPU box.

## Steps

1. **Run the Optuna study.** Launch the TiDE study for the target in the
   background (it is long — ~1 min/trial, so ~1.5 h for 100):

   ```
   TARGET=<t> uv run python -c "import sys; sys.path[:0]=['.','src']; from notebooks.model_training.model import app; app.run()"
   ```

   The study writes to `sqlite:///spp_trials.db`, study name
   `spp_west[_da]_tide`. Monitor progress via the **DB, not the log** (the log is
   progress-bar noise and its Optuna timestamps are **local time** — do not
   compare them to `date -u`):

   ```
   uv run python -c "import optuna; s=optuna.load_study(study_name='<name>', storage='sqlite:///spp_trials.db'); \
   d=[t for t in s.trials if str(t.state)=='TrialState.COMPLETE']; print(len(d),'complete, best', round(s.best_value,3))"
   ```

   Wait for it to reach the trial count (or plateau).

2. **Bake the winners** into `parameters.py` (dry-run first, then write):

   ```
   uv run python scripts/tune_parameters.py --target <t>            # preview
   uv run python scripts/tune_parameters.py --target <t> --write    # apply
   ```

   Then confirm the repo is still healthy: `uv run ruff check src/parameters.py`
   and `uv run pytest tests/unit -q`.

3. **Retrain a candidate** with the new params, staged (not promoted):

   ```
   TARGET=<t> PROMOTE_CHAMPION=false uv run python <retrain runner>
   ```

   (Runner: a 3-line script that sets the env, `load_dotenv`, then
   `from notebooks.model_training.model_retrain import app; app.run()`.) It
   uploads to `models/<t>/retrains/<ts>/` with a `metrics.json` and does NOT
   touch `champion.json`.

4. **Score fairly vs the current champion.** The candidate's `metrics.json` has
   its backtest CRPS + `eval.test_start/test_end`. Because the rolling holdout
   slides, **re-score the current champion on the same window** (load it via
   `utils.download_champion_checkpoints(dir, target='<t>')` and run
   `evaluation.backtest_report` on the same data) rather than trusting stored
   numbers from different dates. Compare CRPS (primary), then coverage/bias as
   guardrails.

5. **Promote only if better.** If the candidate beats the champion's CRPS with
   coverage within tolerance and no bias regression:

   ```
   uv run python scripts/r2_promote_champion.py <ts> --target <t> --promote
   ```

   Otherwise keep the champion and report the candidate's numbers for review.

6. **Report** the top trials, the candidate vs champion CRPS/coverage, and the
   promote decision. If params changed, remind the user to commit
   `src/parameters.py` and (at cutover) redeploy so the scheduled retrain uses
   the tuned params.

## Notes

- `tune_parameters.py` only edits the marked `# >>> TIDE_PARAMS_<TARGET> >>>`
  block; it never touches the other target's params.
- Never compare CRPS across targets (DA and RT are different scales).
- The study is resumable: to add trials, set `REMOVE_PRIOR_MODELS=False` in
  `notebooks/model_training/model.py` and re-run (Optuna continues via the
  sqlite storage). `REMOVE_PRIOR_MODELS=True` starts fresh.
