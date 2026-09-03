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

   Then confirm the repo is still healthy: `uv run ruff check
   src/parameters.py` and `uv run pytest -m "not torch and not e2e" -q`.

3. **Retrain — it self-gates the promotion.** Run the retrain with
   `PROMOTE_CHAMPION=true`:

   ```
   TARGET=<t> PROMOTE_CHAMPION=true uv run python <retrain runner>
   ```

   (Runner: a 3-line script that sets the env, `load_dotenv`, then
   `from notebooks.model_training.model_retrain import app; app.run()`.) The
   retrain trains the candidate, uploads it to `models/<t>/retrains/<ts>/` with a
   `metrics.json`, then runs `evaluation.compare_candidate_to_champion` — a fast
   backtest of the candidate **and** the current champion on the same recent
   window over the fixed `node_list.EVAL_NODES` — and **only promotes if the
   candidate wins on CRPS** (the first champion for a target promotes
   unconditionally). Watch the log for the `Promote gate (<t>): candidate CRPS …
   vs champion … -> PROMOTE / KEEP champion` line.

4. **Report** the top trials and the gate's promote decision. If params
   changed, remind the user to commit `src/parameters.py` and (at cutover)
   redeploy so the scheduled retrain uses the tuned params. To override the
   gate — promote a staged model by hand, or revert — use `uv run python
   scripts/r2_promote_champion.py <ts> --target <t> --promote`.

## Notes

- `tune_parameters.py` only edits the marked `# >>> TIDE_PARAMS_<TARGET> >>>`
  block; it never touches the other target's params.
- Never compare CRPS across targets (DA and RT are different scales).
- The study is resumable: to add trials, set `REMOVE_PRIOR_MODELS=False` in
  `notebooks/model_training/model.py` and re-run (Optuna continues via the
  sqlite storage). `REMOVE_PRIOR_MODELS=True` starts fresh.
