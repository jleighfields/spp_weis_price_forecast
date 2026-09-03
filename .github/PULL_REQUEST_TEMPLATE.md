<!-- Write this so it stands on its own. Your reviewer has the diff and
     nothing else: no plan file, no skill checklist, no memory of the
     conversation that produced this. Say what a reference means instead of
     naming it — "the check that compares the challenger against the current
     champion", not "the promotion gate"; "the retry around the LMP feed", not
     "the fix from yesterday". Numbers and shorthand live inside the file that
     defines them, they shift when it is edited, and half the time that file
     is not in the diff.

     Same for the rest: no "as discussed", no ticket number standing in for
     the reason, no comparison to a version of the code the reviewer cannot
     see. CLAUDE.md's "Comments & docstrings are self-contained" is the same
     rule; this is where it applies to a PR. -->

## Summary
<!-- What does this PR do and why? Include the context needed to judge it. -->

## Changes
<!-- List the main changes in this PR -->
-

## Test plan
- [ ] Tests pass (`uv run pytest`) — the default run, which excludes `torch` and `e2e`
- [ ] Lint clean (`uv run ruff check src tests app.py`)
- [ ] If `src/modeling.py` changed: `uv run pytest -m torch` on a machine with torch
- [ ] If `app.py` changed: `uv run pytest -m e2e` on a machine with torch,
      after `uv run playwright install chromium` (the fixture imports `app.py`,
      which imports torch)

<!-- The `test` check runs the first two, so tick them from a local run and
     let CI be the second opinion rather than the only one. "Tests pass" with
     nothing behind it is the claim CI exists to stop anyone taking on trust.

     The last two are NOT in the required check — the CUDA wheels are about 3.5 GB
     and Playwright needs a browser, so both run after the merge. That means a
     break in either lands first and is noticed second. If your change touches
     what they cover, run them yourself. -->

## Dependencies
<!-- Delete unless pyproject.toml changed. requirements.txt is generated from
     the project dependencies and is what the Modal jobs and the Connect
     deploy install — regenerate it in the same PR, or the deploy and the repo
     disagree. -->
- [ ] `requirements.txt` regenerated
- [ ] `uv.lock` updated (CI runs `uv sync --locked` and fails on a stale lock)

## Manual testing
<!-- Describe any manual testing performed and results -->
