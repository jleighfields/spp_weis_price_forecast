# SPP Price Forecast — Project Instructions

Nodal electricity price forecasting for SPP's western markets: marimo
notebooks + Modal jobs collect SPP marketplace data into R2 (S3-compatible)
storage, `src/` holds the pipeline (collection → engineering → Darts
models), and `app.py` serves forecasts via Shiny, deployed to Posit
Connect. The project is migrating from the retired WEIS market to RTO
West / Integrated Marketplace feeds; in-progress and completed design
plans live under `plans/` (finished ones in `plans/completed/`).

## Skills (Slash Commands)

Custom skills automate the review workflows:

| Command | What it does |
|---------|-------------|
| `/code-quality [file-or-dir]` | Review code for readability, documentation, onboarding, and minimal form (simplification per Minimalism rules) — report-only. (Named to avoid colliding with Claude Code's built-in `/code-review`.) |
| `/comment-docstring <file-or-dir>` | Review and fix docstrings, type hints, inline comments; sweep READMEs for stale prose (edits in place) |
| `/security-scan [file-or-dir]` | Scan for leaked secrets (hardcoded tokens/keys, tracked `.env`/credential files), secret logging, and unsafe defaults (report-only) |
| `/simplify-audit [file-or-dir]` | Repo-wide bloat audit — reports a delete-list of dead code, unused deps, and over-built abstractions (report-only) |

Skills are defined in `.claude/skills/` and committed to the repo.

## Agents

Custom subagents bundle a workflow into a single delegated pass — to
combine multiple steps, or to run a heavy read-only pass in an isolated
context so the main session stays clean:

| Agent | What it does |
|-------|-------------|
| `code-reviewer` | Pre-commit pass: runs the `code-quality` and `security-scan` skills (report-only) then the `comment-docstring` skill (edits in place) over the changed files or a given path |
| `simplify-auditor` | Runs the `simplify-audit` skill in an isolated context and returns a report-only bloat delete-list; keeps the repo-wide grep/read churn out of the main session |

**Run the `code-reviewer` agent before committing non-trivial changes.**
Invoke agents by name, optionally with a file or directory. With no
argument, `code-reviewer` defaults to the changed files
(`git diff --name-only HEAD` plus untracked) and `simplify-auditor`
defaults to the whole repo:

```
> use the code-reviewer agent
> code-reviewer src/data_collection.py
> code-reviewer            # defaults to all changed files
> use the simplify-auditor agent
> simplify-auditor         # whole-repo bloat audit, isolated context
```

Each agent reads its skill's `SKILL.md` at runtime rather than copying
the checklist, so it stays in sync as the skills evolve. Agents are
defined in `.claude/agents/` and committed to the repo. New agent
files are discovered at CLI start, so restart the session after adding
one.

## Minimalism (write less)

An ordered checklist to run *before* writing code. Walk it top to
bottom and stop at the first step that solves the problem:

1. **Does this need to exist?** Prefer not writing it (YAGNI). Deleting
   beats refactoring beats adding.
2. **Use the standard library** before hand-rolling.
3. **Use a library already imported** (`polars`, `pandas`, `numpy`,
   `darts`, `duckdb`, `boto3`, `shiny`) — reach for its built-in before
   writing your own.
4. **Use an already-installed dependency** before adding a new one.
5. **Prefer the smallest correct form.** No helper, class, config knob, or
   generalization until there are 2–3 real call sites — no premature
   abstraction.
6. **Only then** write minimal custom code.

To find existing bloat, run `/simplify-audit` (report-only delete-list).

## Single source of truth for parameter values

Treat parameter values and constants as having exactly one home.
Duplicate-source-of-truth is a class of bug that drifts silently
and is expensive to debug.

**The rule:** every parameter value (a number, a dict, a config
entry) gets ONE home. Other modules read from that home; they
don't redeclare it, copy it, or compile it into a parallel mirror.

**Where parameter values live in this repo:**
- Model + training config (`MODEL_NAME`, `TRAIN_START`, forecast
  horizons, `TIDE_PARAMS` and other hyperparameter dicts) →
  `src/parameters.py`. Notebooks, Modal jobs, and the app import from
  it; they never redeclare the values.
- Storage config (bucket, folder, endpoint) → environment variables
  (`AWS_S3_BUCKET`, `AWS_S3_FOLDER`, `S3_ENDPOINT_URL`,
  `AWS_DEFAULT_REGION`), read via `os.environ` — locally from the
  gitignored `.env`. On Modal the non-secret **bucket name** comes from the
  job's `env={"AWS_S3_BUCKET": ...}` in `modal_jobs/*.py` (so a bucket change
  is a redeploy, not a secret edit); credentials, `S3_ENDPOINT_URL`,
  `AWS_DEFAULT_REGION`, and `AWS_S3_FOLDER` come from the `aws-secret` secret.
  The tracked `.env.example` documents the required keys (and why
  `AWS_S3_FOLDER` must be `""` or end with a trailing slash). The R2 top-level
  layout prefixes have single homes too: `IM_PREFIX`/`WEIS_PREFIX` in
  `src/data_collection_utils.py`, `RETRAINS_PREFIX`/`CHAMPION_KEY_SUFFIX` in
  `src/utils.py`.
- Feed URLs and filename formats → the `get_*_url` builders in
  `src/data_collection.py`; don't paste literal portal URLs elsewhere.

**Common anti-patterns to refuse / fix on sight:**
- A notebook or app module hardcoding a value that `src/parameters.py`
  already defines (training window, quantiles, model name).
- Two dicts in different files that are "supposed to" stay identical.
  Collapse to one dict; restructure imports if a cycle forced the split.
- A function default that silently disagrees with the config constant
  the production path uses.
- A notebook or plot re-computing a value the pipeline already
  produces (e.g., re-aggregating 5-min LMPs instead of calling
  `agg_lmp`). Call the production function; don't fork the math.

## Repo conventions

- **Modal jobs wrap marimo notebooks.** `modal_jobs/*.py` are thin
  wrappers that import a notebook's `app` and call `app.run()`; the
  notebooks under `notebooks/` are the single source of truth for job
  logic. Notebooks are marimo `.py` files (git-friendly, runnable as
  scripts) — edit them as Python, keep cells as separate `@app.cell`
  defs, and put shared logic in `src/`.
- **Updating a Modal job: redeploy in place, never stop + deploy anew.**
  `modal deploy modal_jobs/<job>.py` updates the existing app to a new
  *version* (same app, stays deployed — see `modal app history`); the code
  is baked in from local `src/`/`notebooks/` at deploy time, so redeploy
  after any change. Do **not** `modal app stop` and deploy a renamed app:
  Modal has no way to delete a stopped app, so that leaves permanent
  dashboard clutter. (Ephemeral `modal run` also leaves stopped-app records.)
- **The Databricks jobs in `databricks.yaml` are PAUSED** — Modal
  replaced them. Don't revive them.
- **R2 storage:** WEIS-era data lives under the `weis/` prefix;
  Integrated Marketplace data lands under the `im/` prefix (see the
  migration plan). Keep the WEIS history — it's needed for stitched
  training series.
- **Tests:** `tests/unit` (fast, pure pytest — the default gate:
  `uv run pytest tests/unit -q`) and `tests/e2e` (Playwright driving
  the Shiny app: `uv run pytest tests/e2e -q`; needs
  `uv run playwright install chromium`). Run e2e whenever `app.py` or
  `src/plotting.py` changes.
- **Tooling:** `uv` for envs/commands, `ruff` for lint/format
  (per-file ignores for marimo notebooks live in `pyproject.toml`),
  `detect-secrets` with the committed `.secrets.baseline` for secret
  scanning. `requirements.txt` + `manifest.json` exist for the Posit
  Connect deploy — keep them in sync with `pyproject.toml` when
  runtime deps change.
- **Style:** Google-style docstrings (summary, Args, Returns),
  `X | None` over `Optional[X]`, direct imports for type hints, no `_`
  prefix on function names except internal helpers.
- **Don't reference `plans/` files from code, comments, docstrings, or
  READMEs.** Plans get moved (e.g. to `plans/completed/`), renamed, or
  deleted, which turns any such reference into a broken pointer. Make the
  comment self-contained instead — state the fact/rationale inline rather
  than deferring to a plan. (Plans may reference each other and the code;
  the code just shouldn't reference the plans.)
