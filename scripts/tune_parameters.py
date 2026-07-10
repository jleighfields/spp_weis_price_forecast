"""Bake the top-N Optuna trials for a target into parameters.TIDE_PARAMS_<TARGET>.

Reads a target's TiDE study from the Optuna sqlite DB, takes the top-N complete
trials by CRPS (lower is better), and rewrites the marked TIDE_PARAMS_<TARGET>
block in src/parameters.py. This is the deterministic "update the params"
step of a parameter sweep — no hand-editing of param dicts.

The block is delimited by ``# >>> TIDE_PARAMS_<TARGET> >>>`` /
``# <<< TIDE_PARAMS_<TARGET> <<<`` markers in parameters.py.

Dry-run by default (prints the new block); pass --write to apply.

Usage:
    python scripts/tune_parameters.py --target da            # preview
    python scripts/tune_parameters.py --target da --write    # apply
"""

import argparse
import os
import re
import sys

import optuna

# targets is a darts-free leaf module (no sklearn/darts import needed here).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from targets import DEFAULT_TARGET, TARGETS  # noqa: E402

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "..", "src", "parameters.py")


def build_block(var: str, trials, study_name: str) -> str:
    """Render the TIDE_PARAMS_<TARGET> assignment from the top trials."""
    lines = [
        f"# {var} — top {len(trials)} trials by CRPS from study "
        f"'{study_name}' (CRPS {trials[0].value:.3f}-{trials[-1].value:.3f}). "
        f"Managed by scripts/tune_parameters.py.",
        f"{var} = [",
    ]
    for t in trials:
        lines.append(f"    # trial #{t.number}  CRPS {t.value:.4f}")
        lines.append(f"    {t.params!r},")
    lines.append("]")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=DEFAULT_TARGET, choices=sorted(TARGETS),
                    help=f"forecast target to bake (default {DEFAULT_TARGET!r})")
    ap.add_argument("--top-n", type=int, default=5, help="number of top trials (default 5)")
    ap.add_argument("--model-type", default="tide", help="study model type (default tide)")
    ap.add_argument("--storage", default="sqlite:///spp_trials.db",
                    help="Optuna storage URL (default sqlite:///spp_trials.db)")
    ap.add_argument("--write", action="store_true",
                    help="write the block to parameters.py (default: dry run)")
    args = ap.parse_args()

    study_name = f"{TARGETS[args.target]['model_name']}_{args.model_type}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=args.storage)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print(f"no complete trials in study {study_name!r}; nothing to bake")
        return 1
    if len(complete) < args.top_n:
        print(f"WARNING: only {len(complete)} complete trials (< top-n {args.top_n})")
    top = sorted(complete, key=lambda t: t.value)[: args.top_n]

    var = f"TIDE_PARAMS_{args.target.upper()}"
    block = build_block(var, top, study_name)

    print(f"study {study_name!r}: {len(complete)} complete trials; top {len(top)} by CRPS:")
    for t in top:
        print(f"  trial #{t.number}  CRPS {t.value:.4f}")
    print("\n--- new block ---\n" + block + "\n")

    with open(PARAMS_FILE) as f:
        src = f.read()
    begin, end = f"# >>> {var} >>>", f"# <<< {var} <<<"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(src):
        print(f"ERROR: markers {begin} / {end} not found in {PARAMS_FILE}")
        return 1
    new_src = pattern.sub(f"{begin}\n{block}\n{end}", src)

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
