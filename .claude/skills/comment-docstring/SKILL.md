---
name: comment-docstring
description: Review Python files for missing docstrings, type hints, and inline comments. Generate Google-style docstrings, suggest helpful comments for onboarding, and sweep READMEs for stale references the changes invalidate.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Edit, Bash
argument-hint: <file-or-directory>
---

# Comment & Docstring Review

Scan Python files for missing or incomplete documentation. Sweep
nearby READMEs for stale prose the changes invalidate. Fix issues
in place and report what was changed.

## Arguments

- **file-or-directory** (required): Path to a `.py` file or directory
  to scan. If a directory, scan all `.py` files recursively.

## What to Check

For each function and class:

1. **Docstring exists** — every public function needs one
2. **Google style** — summary line, Args section, Returns section
3. **Type hints** — all parameters and return types annotated
4. **Summary is accurate** — matches what the function actually does
5. **Parameters documented** — every parameter listed with type and description
6. **Returns documented** — return type and meaning described

For the file body:

7. **Inline comments** — add comments that explain *why*, not *what*
8. **Section headers** — use `# --- Section Name ---` for logical blocks
   (this repo also uses `#####`-bar section headers in `src/` — match
   the style already present in the file)
9. **Magic numbers** — explain any hardcoded values
10. **Complex logic** — add comments for non-obvious algorithms

## Style Rules

Follow the project's CLAUDE.md conventions:

- Google-style docstrings with summary, Args, and Returns
- Use `X | None` syntax (not `Optional[X]`)
- Import modules directly for type hints (no forward references)
- Prioritize simplicity and readability
- Add comments that help someone on-boarding to the project
- Do not prepend function names with `_` (except internal helpers)

## README sweep

Every pass under this skill should also sweep the repo's prose docs
for content that the diff just invalidated — stale formulas, removed
function/column names, retired flags, outdated tables. README drift is
a common silent-failure mode: the code moves, the doc says it didn't,
and the next developer follows the README into the wrong mental model.

### Which docs to check

| Touched file under… | Sweep these docs |
|---|---|
| `src/`, `app.py`, `modal_jobs/`, `notebooks/` | root `README.md` |
| `scripts/` | `scripts/README.md` |
| Anything renaming feeds, storage prefixes, model names, or node scope | `plans/weis_to_rto_west_migration.md` (the live migration plan) |
| Anything that changes repo-level architecture, conventions, or shared SSoT locations | `CLAUDE.md` (root) and `README.md` (root) |

### What to look for

Run a targeted grep over the doc set with the names of every
removed / renamed / refactored symbol from the diff. Common shapes
of stale prose to flag:

- **Removed constants or functions** — e.g., a README citing a deleted
  `get_*_url` builder. Update the prose to point at the surviving
  mechanism (or drop the bullet entirely).
- **Renamed columns or files** — e.g., a WEIS-era filename prefix or
  column name that the Integrated Marketplace migration renamed.
  Tables and bullet lists that reference the old name by hand need a
  sweep.
- **Stale commands** — run/test/deploy instructions that no longer
  match the Modal jobs, marimo notebooks, or pytest layout.
- **Stale config references** — prose citing a `src/parameters.py`
  constant that was renamed or removed.

### What NOT to do

- **Don't add dates or "removed in" history** to READMEs. Prose
  describes the current state; git log carries the history.
  "The WEIS collector was retired 2026-04-01" is the wrong style —
  write "collection targets the Integrated Marketplace feeds" and let
  `git blame` answer "since when?".
- Don't write a "Migration notes" / "Changelog" section in a README
  to track a refactor — status lives in the plan under `plans/`.

## Example Output

```python
def prep_lmp(
    con: duckdb.DuckDBPyConnection,
    start_time: pd.Timestamp | None = None,
    loc_filter: str = 'PSCO_',
) -> pl.DataFrame:
    """Load and filter LMP data for model training.

    Reads the consolidated LMP table, restricts it to the training
    window and the settlement locations matching ``loc_filter``, and
    derives the ``unique_id`` / ``timestamp_mst`` columns the Darts
    pipeline expects.

    Args:
        con: DuckDB connection with the 'lmp' table loaded.
        start_time: Start of time range filter. If None, uses
            parameters.TRAIN_START relative to now.
        loc_filter: Pattern to filter Settlement_Location_Name.

    Returns:
        LMP data with 'unique_id', 'timestamp_mst', and price columns.
    """
```

## Steps

1. Read the target file(s)
2. For each function/class, check docstring completeness
3. For each function signature, check type hints
4. Scan for places where inline comments would help
5. Apply the "README sweep" section above — pick the doc set from the
   matching-files table, grep for every removed / renamed symbol from
   the diff, and fix any prose the change invalidated.
6. Make edits directly (don't just report — fix)
7. Run `uv run pytest tests/unit -q` to verify nothing broke
8. Report a summary of changes made (including any README / plan
   updates)
