# SPP Price Forecast — Project Instructions

Nodal electricity price forecasting for SPP's western markets: marimo
notebooks + Modal jobs collect SPP marketplace data into R2 (S3-compatible)
storage, `src/` holds the pipeline (collection → engineering → Darts
models), and `app.py` serves forecasts via Shiny, deployed to Posit
Connect. The project is migrating from the retired WEIS market to RTO
West / Integrated Marketplace feeds; in-progress and completed design
plans live under `plans/` (finished ones in `plans/completed/`).

## Skills and agents

Six review skills live in `.claude/skills/` and two subagents in
`.claude/agents/`; `.claude/README.md` indexes them, gives each agent's default
target, records which file owns which kind of rule, and lists where the shared
facts live. Their names and descriptions are injected at session start, so none
is listed here.

**Run `code-reviewer commit` before committing non-trivial changes**, and the
full pass — `code-reviewer` with no argument — before opening a pull request.
With no argument it reviews the branch diff against `origin/main`; `commit`
reviews the changed files.

## Review in two tiers

Both tiers are run by the `code-reviewer` agent, and the difference is what
each is defined over:

- **`commit`, per commit, over what is being committed.** The three
  report-and-fix skills, and nothing else. It does **not** run the suite —
  you run the tests the commit can reach.
- **The full pass, per branch, before its pull request opens.** Adds the
  suite over the whole change, the `test-review` mutation phase, and a
  failing test pinning any confirmed defect.

**Resolve or waive every Must Fix and Should Fix before opening the pull
request.** Never let a finding lapse by calling it "pre-existing" or "out of
scope" — surface it for an explicit decision.

`main` is protected and takes no direct pushes, so every change arrives
through a squash-merged pull request. The required `test` check runs the
default marker set, `-m "not torch and not e2e"`. That set omits torch and the
CUDA stack, so `src/modeling.py` is not exercised on the gate; the `torch` and
`e2e` markers run after the merge in `heavy.yml`. Get the current split with
`uv run pytest --collect-only -q` rather than trusting a count written here.

## Prose is professional and factual

**Everything written here — comments, docstrings, READMEs, plans, commit
messages, pull-request bodies, skills and agents — states what is true and
how the reader can check it.** A sentence that rates something without
evidence describes the author's opinion, not the code's behavior. When the
code changes, unsupported ratings do not update with it.

Common categories to avoid: unmeasured rankings, personified programs where
the verb stands in for a mechanism, unmeasured cost or effort claims,
aesthetic verdicts like "elegant" or "hacky", aphorisms, and filler run-ups.
See `comment-docstring` for rewrites, greps, and the categories that need
manual review.

**A model-performance claim is a measurement or it is nothing.** Name the
metric, the evaluation window, and the nodes it was scored over. "The new
model is better" is the exact sentence this section exists to prevent —
`compare_candidate_to_champion` produces the numbers, so quote them.

**Argument is not editorializing.** State each claim with its reason, in the
same sentence or the next one — for example, "two copies of the same value
drift apart over time." Give the reader something to check; keep the
reasoning and drop unsupported ratings.

Judge sentences in context — some individual words that look like offenders
are fine. See `comment-docstring` for details.

## Comments & docstrings are self-contained

**Every comment, docstring, marimo cell, and doc must stand on its own for a
reader who has the repo and nothing else**, and must describe the code as it
is now. References that only make sense outside the repo, or only to people
involved in the original conversation, break for future readers.

Common categories to avoid: references to commits, tickets, "as discussed",
earlier versions of the code, shortened domain terms that collapse to common
English words, and bare dates. See `comment-docstring` for examples and
greps.

Describe the thing directly — what it does, what the constraint is, why this
way rather than the obvious alternative. Test: **delete every ticket and
commit message; would this sentence still teach a new reader anything?**

Point to durable references freely: a README section, another module, an
external spec. **Not a plan under `plans/`** — this repo's convention is that
code, comments, docstrings and READMEs do not reference plan files, and that
rule wins here. Ask whether the reference will still exist a year from now: a
README section will, a ticket number may not.

- **Pull-request and issue bodies, at a stricter bar.** Their reader has the
  diff and little else, so even a pointer into this repo fails when the diff
  omits the file it points at. Name the thing, not its number.
  `.github/PULL_REQUEST_TEMPLATE.md` carries this reminder at the point of
  writing.
- **Directory READMEs point, never restate.** Each says what belongs in its
  directory and links to whatever owns the detail. Duplicated descriptions go
  stale when the code moves.

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
- Forecast **target** dimension → `parameters.TARGETS` (`'da'` day-ahead,
  the `DEFAULT_TARGET`/primary; `'rt'` real-time, parked) + `DEFAULT_TARGET`.
  Each target trains from its own source parquet (`create_database(target=…)`
  loads `im/lmp.parquet` or `im/da_lmp.parquet`) into its own model namespace
  `models/<target>/` (via `utils.retrains_prefix(target)` /
  `champion_key_suffix(target)`). The retrain notebook + Modal jobs pick the
  target from the `TARGET` env var; the app picks it from the `input.target`
  selector. Never compare metrics across targets.
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
- Feed URLs and filename formats → the `get_*_url` builders:
  `src/data_collection_im.py` for the Integrated Marketplace feeds,
  `src/data_collection.py` for the retired WEIS ones. Don't paste literal
  portal URLs elsewhere.

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
- **The Databricks jobs in `deprecated/databricks.yaml` are PAUSED** — Modal
  replaced them. Don't revive them.
- **R2 storage:** WEIS-era data lives under the `weis/` prefix;
  Integrated Marketplace data lands under the `im/` prefix. Keep the WEIS
  history — the stitched training series needs it.
- **Tests:** `tests/unit` (fast, pure pytest — the default gate:
  `uv run pytest -m "not torch and not e2e" -q`) and `tests/e2e` (Playwright driving
  the Shiny app: `uv run pytest -m e2e -q`; needs
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
  prefix on function names — internal helpers get real names too.
- **Don't reference `plans/` files from code, comments, docstrings, or
  READMEs.** Plans get moved (e.g. to `plans/completed/`), renamed, or
  deleted, which turns any such reference into a broken pointer. Make the
  comment self-contained instead — state the fact/rationale inline rather
  than deferring to a plan. (Plans may reference each other and the code;
  the code just shouldn't reference the plans.)
