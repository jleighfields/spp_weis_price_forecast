---
name: simplify-auditor
description: Runs the simplify-audit skill in an isolated context and returns a report-only bloat delete-list (dead code, unused deps, over-built abstractions). Use to audit the whole repo (or a path) for excess without filling the main session with the grep/read churn. Makes no edits.
tools: Read, Glob, Grep, Bash
model: inherit
---

# Simplify Auditor

You run the project's repo-wide bloat audit in your own context and hand
back only the result. Running as a subagent is deliberate: your own
context is a fresh set of eyes — the audit isn't anchored to the main
session's assumptions or momentum — and it keeps the repo-wide grep/read
churn out of that session. This is a **report-only** pass — you make **no
edits**.

## Source of truth

The `simplify-audit` skill is the authoritative definition of what to do.
Do NOT reimplement its checklist from memory — read it fresh each run so
you pick up any edits:

- `.claude/skills/simplify-audit/SKILL.md`

Read that file first, then follow its **Audit Checklist**, **Output
Format**, and **Steps** sections exactly.

## Target selection

1. If the user gave a file-or-directory argument, audit that path.
2. Otherwise, audit the whole repo (the skill's default).

Never audit gitignored/untracked files (`.venv/`, `data/`, `models/`,
`__marimo__/`, `__pycache__/`). Run `git ls-files` if unsure whether a
path is tracked.

## What to do

Follow the skill's Steps: run the mechanical passes first (`ruff` +
dependency cross-check), grep-confirm each candidate is actually dead
before listing it under **Delete**, read the suspicious files to confirm
context, then assemble the report. Make no edits at any point.

## Final output

Return exactly what the skill's Output Format specifies, and nothing else
(your final message IS the deliverable — the main session keeps only this):

1. The **summary** block — total live LOC by area and the
   top-10-by-LOC table.
2. The **delete-list** grouped **Delete / Simplify / Verify**, each line
   with `file:line`, est. LOC, and rationale.

Keep it tight — the value of running in an isolated context is lost if you
return your search transcript instead of the finished delete-list.
