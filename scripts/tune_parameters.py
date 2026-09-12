"""Bake the top-N Optuna trials for a target into parameters.TIDE_PARAMS_<TARGET>.

Reads a target's TiDE study from the Optuna sqlite DB, takes the top-N complete
trials by the objective mode's composite score (lower is better), and rewrites
the marked TIDE_PARAMS_<TARGET> block in src/parameters.py. This is the
deterministic "update the params" step of a parameter sweep — no hand-editing
of param dicts.

The objective mode (src/selection.py) selects both the study to read and the
formula the trials are ranked by, so a bake can never rank one mode's trials by
another mode's formula. It is part of the study name for the same reason.

The block is delimited by ``# >>> TIDE_PARAMS_<TARGET> >>>`` /
``# <<< TIDE_PARAMS_<TARGET> <<<`` markers in parameters.py.

Dry-run by default (prints the new block); pass --write to apply.

Usage:
    python scripts/tune_parameters.py --target da            # preview
    python scripts/tune_parameters.py --target da --write    # apply
    python scripts/tune_parameters.py --target da --objective crps   # other mode
"""

import argparse
import os
import re
import sys

import optuna

# targets and selection are darts-free leaf modules (no sklearn/darts import
# needed here).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import selection  # noqa: E402
from targets import DEFAULT_TARGET, TARGETS  # noqa: E402

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "..", "src", "parameters.py")


def trial_summary(metrics: dict, mode: dict) -> str:
    """One-line ``METRIC value`` summary of a trial under the ranking mode.

    Reads the recombined metric dict rather than the raw objective values, so
    the numbers shown are the ones the *requested* mode scores on — not
    whatever weighting the sweep happened to run under.

    Args:
        metrics: A trial's metrics by name, from ``selection.trial_metrics``.
        mode: The ``OBJECTIVES`` entry being ranked under.

    Returns:
        A space-separated line, e.g. ``"MAE 6.0912  CI_ERR 1.5000  score
        7.5912"``.
    """
    parts = []
    for name in mode["metrics"]:
        if name == "ci_err":
            errors = {
                b: metrics[selection.band_label(b)] for b in mode["intervals"]
            }
            parts.append(f"CI_ERR {selection.weight_ci_errs(errors, mode):.4f}")
        else:
            parts.append(f"{name.upper()} {metrics[name]:.4f}")
    parts.append(f"score {selection.score_metrics(metrics, mode):.4f}")
    return "  ".join(parts)


def build_block(
    var: str,
    trials: list[optuna.trial.FrozenTrial],
    study_name: str,
    mode_name: str,
    mode: dict,
    metrics: dict[int, dict],
) -> str:
    """Render the ``TIDE_PARAMS_<TARGET>`` assignment source from the top trials.

    Args:
        var: The target variable name, e.g. ``"TIDE_PARAMS_DA"``.
        trials: The top-N trials, already sorted best (lowest composite score)
            first; each contributes one ``{...}`` param dict to the list.
        study_name: Optuna study name, recorded in the block's header comment.
        mode_name: The objective mode the trials were ranked under.
        mode: That mode's config from ``selection.OBJECTIVES``.
        metrics: Per-trial metric dicts keyed by trial number, from
            ``selection.trial_metrics`` — scored under ``mode`` here, which may
            differ from the mode the study was swept under.

    Returns:
        The Python source for the assignment — a header comment plus the list
        of param dicts (each preceded by a ``# trial #N ...`` comment naming the
        metrics and composite score it was picked on).
    """
    scores = [selection.score_metrics(metrics[t.number], mode) for t in trials]
    lines = [
        f"# {var} — top {len(trials)} trials by '{mode_name}' score from study "
        f"'{study_name}' (score {scores[0]:.3f}-{scores[-1]:.3f}; "
        f"metrics {'+'.join(mode['metrics'])}). "
        f"Managed by scripts/tune_parameters.py.",
        f"{var} = [",
    ]
    for t in trials:
        lines.append(f"    # trial #{t.number}  {trial_summary(metrics[t.number], mode)}")
        lines.append(f"    {t.params!r},")
    lines.append("]")
    return "\n".join(lines)


def main() -> int:
    """Bake a target's top-N Optuna trials into parameters.py (or dry-run).

    Returns:
        Process exit code: 0 on success (or dry run), 1 if the study has no
        complete trials or the parameters.py markers are missing.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=DEFAULT_TARGET, choices=sorted(TARGETS),
                    help=f"forecast target to bake (default {DEFAULT_TARGET!r})")
    ap.add_argument("--top-n", type=int, default=selection.TOP_N,
                    help=f"number of top trials (default {selection.TOP_N}, "
                         "the ensemble size the retrain slices to)")
    ap.add_argument("--model-type", default="tide", help="study model type (default tide)")
    ap.add_argument("--storage", default="sqlite:///spp_trials.db",
                    help="Optuna storage URL (default sqlite:///spp_trials.db)")
    ap.add_argument("--objective", default=selection.DEFAULT_OBJECTIVE,
                    choices=sorted(selection.OBJECTIVES),
                    help="objective mode the study was run under and is ranked "
                         f"by (default {selection.DEFAULT_OBJECTIVE!r})")
    ap.add_argument("--write", action="store_true",
                    help="write the block to parameters.py (default: dry run)")
    args = ap.parse_args()

    mode = selection.resolve_mode(args.objective)
    study_name = selection.study_name(
        TARGETS[args.target]["model_name"], args.model_type, args.objective
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=args.storage)

    # A study stored with a different objective count cannot be ranked under
    # this mode: trial.values would have the wrong arity, and reading trial.value
    # on a multi-objective study raises. Fail here with the mismatch named
    # rather than deep inside the ranking comprehension.
    if len(study.directions) != len(mode["metrics"]):
        print(
            f"ERROR: study {study_name!r} has {len(study.directions)} objective(s) "
            f"but mode {args.objective!r} expects {len(mode['metrics'])} "
            f"{mode['metrics']}; re-run the study under this mode."
        )
        return 1

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print(f"no complete trials in study {study_name!r}; nothing to bake")
        return 1
    if len(complete) < args.top_n:
        print(f"WARNING: only {len(complete)} complete trials (< top-n {args.top_n})")

    # A trial's objective values only mean something against the mode that
    # produced them: values[1] is a ci_err already weighted by the SWEEP's
    # scalers. Ranking those sums under a different mode would silently apply
    # the sweep's weights while reporting the requested mode's name, so
    # recombine each trial's metrics and recompute the calibration term.
    swept_name = study.user_attrs.get("objective_mode")
    if swept_name is None:
        print(
            f"ERROR: study {study_name!r} does not record the objective mode it "
            f"was run under, so its trials cannot be re-ranked safely. Set it "
            f"with: optuna.load_study(...).set_user_attr('objective_mode', <mode>)"
        )
        return 1
    swept_mode = selection.resolve_mode(swept_name)
    if swept_name != args.objective:
        print(
            f"NOTE: study was swept under {swept_name!r}; re-ranking under "
            f"{args.objective!r} (calibration term recomputed from per-band errors)."
        )

    metrics = {
        t.number: selection.trial_metrics(t.values, t.user_attrs, swept_mode)
        for t in complete
    }
    top = sorted(
        complete, key=lambda t: selection.score_metrics(metrics[t.number], mode)
    )[: args.top_n]

    var = f"TIDE_PARAMS_{args.target.upper()}"
    block = build_block(var, top, study_name, args.objective, mode, metrics)

    print(
        f"study {study_name!r}: {len(complete)} complete trials; "
        f"top {len(top)} by {args.objective!r} score:"
    )
    for t in top:
        print(f"  trial #{t.number}  {trial_summary(metrics[t.number], mode)}")
    print("\n--- new block ---\n" + block + "\n")

    with open(PARAMS_FILE) as f:
        src = f.read()
    begin, end = f"# >>> {var} >>>", f"# <<< {var} <<<"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(src):
        print(f"ERROR: markers {begin} / {end} not found in {PARAMS_FILE}")
        return 1
    # Function replacement so `block` (which contains repr'd param values) is
    # inserted literally — a plain string replacement would interpret any
    # backslash / \g<> / \1 in it. count=1 documents the single-block intent.
    replacement = f"{begin}\n{block}\n{end}"
    new_src = pattern.sub(lambda _m: replacement, src, count=1)

    if not args.write:
        print("[dry-run] pass --write to update parameters.py")
        return 0
    with open(PARAMS_FILE, "w") as f:
        f.write(new_src)
    print(f"WROTE {var} ({len(top)} trials) to {PARAMS_FILE}")
    print("Re-train the champion (TARGET=<t>) to use the new params, then promote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
