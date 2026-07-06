---
name: code-quality
description: Review code for readability, documentation quality, onboarding ease, and minimal form (simplification per the project Minimalism rules). Report-only — presents findings, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [file-or-directory]
---

# Code Quality Review

Review code for readability, documentation quality, ease of
on-boarding, and whether the code is in its minimal form
(simplification per the project Minimalism rules). Report issues but do
NOT make edits — present findings for the user to approve.

(Named `code-quality` rather than `code-review` to avoid colliding with
Claude Code's built-in `/code-review` command.)

## Arguments

- **file-or-directory** (optional): Path to review. If omitted, review
  all staged and unstaged changed files (`git diff --name-only HEAD`).

## Review Checklist

### Readability

1. **Function length** — flag functions > 50 lines. Can they be split?
2. **Variable names** — are they descriptive? Flag single-letter vars
   (except `i` for loop index; short-lived `df` is acceptable)
3. **Nesting depth** — flag > 3 levels of nesting. Can early returns
   or guard clauses simplify?
4. **Magic numbers** — flag hardcoded values without explanation
5. **Dead code** — commented-out code, unused imports, unreachable branches
6. **Silent failures** — missing file/dir checks that skip without logging a warning
7. **Arbitrary decisions** — thresholds, caps, multipliers, or logic
   branches with no comment explaining *why* that value or approach was
   chosen (e.g., a `-7h` timezone offset or a retry cap with no
   justification)

### Documentation

8. **Missing docstrings** — every public function and class needs one
9. **Outdated docstrings** — does the docstring match the current code?
10. **Missing type hints** — all function signatures need types
11. **Confusing comments** — comments that describe *what* instead of *why*
12. **Missing comments** — complex logic without explanation

### Style (per CLAUDE.md)

13. **Google-style docstrings** with summary, Args, Returns
14. **`X | None`** syntax (not `Optional[X]`)
15. **Direct imports** for type hints (no forward references)
16. **No `_` prefix** on function names (except internal helpers)

### Code Duplication & Helper Functions

Flag repeated patterns and recommend concrete extractions. This is
a **Should Fix** at 2 copies and a **Must Fix** at 3+.

17. **Near-identical functions** — two or more functions that share
    >50% of their logic with only minor parameter differences (e.g.,
    different URL slugs, different column names). Extract the shared
    body into a parameterized helper and make the public functions thin
    wrappers. The `get_process_mtlf` / `get_process_mtrf` /
    `get_process_5min_lmp` family in `src/data_collection.py` is the
    canonical local example of the shape to watch.
18. **Repeated multi-line patterns in notebooks** — if the same 3+
    line sequence appears in multiple cells, extract it into a
    notebook-local helper function or a module-level function in `src/`.
    Marimo cells must remain separate `@app.cell` defs for the reactive
    graph, but the *body* of each cell can call a shared helper.
19. **Copy-pasted logic with small variations** — loops, conditions,
    or data-processing blocks that were clearly copied and tweaked
    (e.g., the same read → format → filter → write pattern for two
    feeds). Extract into a function parameterized on the varying parts.
20. **Hardcoded values repeated across files** — the same magic number
    or string literal appearing in 2+ files without a shared constant.
    Extract to `src/parameters.py` or a module-level constant.
21. **When NOT to extract** — do not flag single-use patterns shorter
    than 3 lines, marimo cell signatures (they must list dependencies
    explicitly), or test setup code (test clarity > DRY).

When flagging duplication, always include:
- Which functions/blocks are duplicated
- How many copies exist
- A concrete suggested helper signature (name, parameters, return type)

For the broader minimalism lens (code that shouldn't exist, reinvention
of built-ins, premature abstraction), see **Simplification** below
(items 31–35).

### Single Source of Truth for Parameter Values

A high-priority class of review findings. Parameter values (numbers,
dicts, config entries) get ONE home; other modules read from it.
Duplicate-source-of-truth bugs drift silently and are expensive to
debug. Flag every instance, with severity calibrated to blast radius:

- **Must Fix** when the duplication is in the production collection /
  modeling path (`src/`, `modal_jobs/`) or when the copies have
  already drifted (different values).
- **Should Fix** when the duplication is in plots, notebooks, or tests
  but the values currently agree (drift is a matter of time).

See CLAUDE.md §"Single source of truth for parameter values" for
where canonical values live in this repo.

22. **Hardcoded scalar where a `src/parameters.py` constant exists** —
    e.g., a notebook or app module re-declaring the training window,
    forecast horizon, quantile list, or model name instead of importing
    from `src.parameters`. Two sources for the same value drift apart
    on the next edit. Fix: import the constant; remove the hardcode.
    Specifically watch for: `TRAIN_START`, `MODEL_NAME`, forecast
    horizons, model hyperparameter dicts, node/location filters.
23. **Duplicated config dicts kept in sync by hand or by a compile
    step** — a dict in `src/parameters.py` and a parallel mirror in a
    notebook or app module that are "supposed to" stay identical. The
    dup is the bug; a sync script or startup assertion is a band-aid.
    **Must Fix: collapse to one dict** — if an import cycle forced the
    split, restructure imports (move shared constants to a leaf module).
24. **Mismatched defaults across functions for the same logical
    value** — e.g., a function whose `start_time=None` default resolves
    differently from `parameters.TRAIN_START`. Caller omits the kwarg →
    silent behavior split between tests and production. Fix: align
    defaults, or require the parameter (no default).
25. **Function reimplements a calculation the production code already
    does** — e.g., a notebook re-aggregating 5-min LMPs by hand when
    `src.data_collection.agg_lmp` is the production path, or the app
    re-deriving `timestamp_mst`. Even small differences compound. Fix:
    call the production function; don't fork the math.
26. **App and pipeline compute the same value independently** — e.g.,
    the app rebuilding the settlement-location universe with its own
    query while data engineering derives it from the LMP data. Cache or
    read the pipeline's output; the code that wrote the canonical value
    stays the sole computer.

When flagging a config-source violation, include:
- The two (or more) source locations
- Whether they currently agree (same value, just two copies) or
  already drift (different values)
- Which one is canonical (`src/` production code usually wins)
- The signature change required to consolidate

### On-boarding Ease

27. **Would a new team member understand this?** — flag sections that
    need context comments
28. **Are error messages helpful?** — do assertions explain what went wrong?
29. **Are log messages informative?** — do they help debug a failed
    collection or retrain run?
30. **Are READMEs up to date?** — do they reflect the current code?

### Simplification (per Minimalism rules)

Apply the **Minimalism (write less)** hierarchy from `CLAUDE.md` to the
changed code: walk it top to bottom and flag where the diff skipped a
cheaper step. This lens is about the *form* of the new code (is it
minimal?), as distinct from **Duplication** (items 17–21, is it repeated?),
the **Single Source of Truth** section (items 22–26, is a config value
duplicated?), and the `simplify-audit` skill (is there dead/excess code
across the *whole repo*?). Report only — do not edit.

31. **Existence / YAGNI** — does the new code need to exist? Flag
    speculative helpers, unused parameters, config knobs, or branches the
    diff adds "just in case" with no current caller.
32. **Reinvention** — hand-rolled logic that duplicates a built-in from the
    stdlib or an already-imported library (`polars`, `pandas`, `numpy`,
    `darts`, `duckdb`, `boto3`, `shiny`). Cite the built-in that replaces
    it (e.g. a manual accumulation loop that
    `df.group_by(...).agg(...)` does in one line).
33. **Single-use abstraction** — a wrapper, helper, or class the diff
    introduces for exactly one call site. Recommend inlining. This is the
    inverse of items 17–21: extract at 2–3 copies, inline at one.
34. **Premature generalization** — parameters, `**kwargs`, or branches that
    handle cases which do not occur in the codebase yet.
35. **Smallest correct form** — multi-line constructs that collapse to a
    comprehension, vectorized op, or single call; needless intermediate
    variables.

For each simplification finding, cite the Minimalism step (1–6) it maps
to so the user sees which rule applies.

## Output Format

Group findings by severity. **Number findings sequentially across
all three buckets** (1, 2, 3, ...) starting at 1 in Must Fix and
continuing through Should Fix and Consider. Sequential numbering
gives every finding a unique short ID the user can reference in
conversation ("apply 1, 4, 7", "skip #11").

### Must Fix
- Critical issues (wrong docstrings, misleading comments, missing types)

### Should Fix
- Important readability issues (long functions, missing comments)

### Consider
- Style suggestions, minor improvements

For each finding, include:
- A leading sequential number (continuing from prior bucket)
- File and line number
- What the issue is
- Suggested fix (brief)

Example layout:

```
## Must Fix

### 1. `path/to/file.py:42` — function raises but lacks Raises: section
...

## Should Fix

### 2. `path/to/file.py:100-150` — function is 80 lines, split into ...
...

### 3. `path/to/other.py:5` — duplicated logic with file.py:42
...

## Consider

### 4. `path/to/file.py:60` — variable name `x` could be `node_count`
...
```

## Steps

1. Identify files to review:
   - If a file or directory argument is provided, review those files
   - **If no argument is provided**, you MUST run this command to find
     all changed files and review every one of them:
     ```bash
     git diff --name-only HEAD
     ```
     If this returns no files, also check for untracked files:
     ```bash
     git status --short
     ```
     Review ALL files returned — do not skip any.
2. Run static checks on the changed files first — these surface
   issues mechanically before you start reading:
   - `uv run ruff check <changed-files>` — unused imports, undefined
     names, style violations
   - `uv run ruff format --check <changed-files>` — formatter drift
   - `uv run pytest tests/unit -q` — the fast unit suite
   - **If any changed file touches the app surface** (`app.py`,
     `src/plotting.py`, or anything under `tests/e2e/`), also run the
     Playwright end-to-end suite: `uv run pytest tests/e2e -q`. The
     unit suite does NOT exercise the rendered app, so render bugs an
     `import`/unit pass can't see only surface by driving a real
     browser. Report a failure as **Must Fix**. Needs Playwright
     browsers (`uv run playwright install chromium`); if they can't
     launch, say so and recommend the user run it rather than silently
     skipping.

   Treat any ruff finding as **at least Should Fix**; F821 (undefined
   name) and most B-class rules are **Must Fix** since they're real
   bugs. Cite the rule code (e.g. `F401`, `B008`) in each finding so
   the user knows what `--fix` would do. Skip checklist items 5
   (dead code/unused imports) and 15 (forward references) — ruff
   covers them more reliably than human review. **Do not report `S`
   (flake8-bandit) findings here** — those are security concerns owned
   by the `security-scan` skill (the `code-reviewer` agent runs it as a
   separate phase); reporting them here would double-count.

   Note on marimo notebooks: ruff is configured (pyproject.toml
   `[tool.ruff.lint.per-file-ignores]`) to ignore B018, E501, F401,
   F811, F821, I001, and S101 under `notebooks/**/*.py` because
   marimo's reactive graph violates ruff's normal expectations
   (bare expressions for output, cross-cell imports). Don't manually
   re-flag these in notebook files.
3. Read each file fully — do not skip any changed files
4. Apply the checklist to every changed file, paying special attention
   to code duplication (items 17-21) and simplification (items 31-35).
   For each duplication finding, include a concrete helper signature so
   the fix is actionable; for each simplification finding, cite the
   Minimalism step (1-6) it maps to.
5. Present findings grouped by severity, ruff findings cited inline
6. Do NOT make edits — let the user decide what to fix
7. After presenting findings, suggest running ``/comment-docstring``
   on the changed files to fill any docstring / type-hint / inline-
   comment gaps. Do not invoke it automatically — that's a separate
   editing step the user should opt into.

## After the review: present, then gate

This skill is report-only: **present every finding (numbered) for the
user to review — never auto-fix.** But the findings are a tracked
checklist, not advisory prose:

- Treat each **Must Fix** and **Should Fix** as an open item referenced
  by its number. Do **not** commit or merge the reviewed code until every
  such item is either fixed or **explicitly waived by the user**.
- "Pre-existing" / "out of scope" is never a silent pass — name the
  finding and get the user's waiver; do not drop it by omission.
- **Consider** items are optional and do not block.
