---
name: security-scan
description: Scan for leaked secrets and insecure patterns — hardcoded passwords/tokens/keys, tracked .env or credential files, secrets in logs, and unsafe defaults. Drives ruff S + detect-secrets with a grep fallback; report-only, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: "[file-or-directory]"
---
# Security Scan

Find secrets and unsafe patterns before they get committed, and report a
findings list. **Report-only** — never edit or delete a secret yourself.
Removing a secret and (critically) **rotating** it is a deliberate,
human-driven step.

Run tooling first, the same way `code-quality-review` runs ruff, then cover
the remaining cases with `git` checks and `grep`:

1. **`ruff` `S` rules (flake8-bandit)** — mechanical insecure-pattern
   detection (hardcoded passwords, `eval`/`exec`, `shell=True`, unsafe
   deserialization). Configured in `pyproject.toml`.
2. **`detect-secrets`** — entropy + regex secret detection with a committed
   `.secrets.baseline` so known false positives stay suppressed.
3. **`git ls-files` + `grep`** — tracked-credential-file checks and a regex
   fallback for token shapes the tools miss.

Security scan findings answer "does this leak a credential or open a hole?"
`code-quality-review` answers "is it clear?" and `simplify-audit` answers
"should it exist?"

## Arguments

- **file-or-directory** (optional): Path to scan for secret *content* /
  insecure patterns. If omitted, scan the changed files
  (`git diff --name-only HEAD` unioned with `git ls-files --others
  --exclude-standard` — union, not fallback).
  The tracked-file checks always run against the whole repo regardless of
  the argument.

## Scope

- **Content / pattern scan:** the target files (changed files by default,
  or the given path).
- **Tracked-file checks:** always whole-repo via `git ls-files` — a
  committed `.env` is a repo-wide fact, not a diff fact.
- **Never scan gitignored or untracked artifacts** for content. Confirm
  with `git ls-files` rather than guessing from directory names — but DO
  still confirm those paths are actually ignored.

## What to check

### Tracked credential files (whole repo, always)

Run `git ls-files` and flag any tracked file that should never be committed:

- `.env`, `.env.local`, `.env.*` **except** `.env.example` / `.env.template`
  / `.env.sample` (those are intended templates — see the caveat below).
- `.modal.toml` and `rsconnect-python/` config directories. Both hold a
  working token once written, and both arrive with a `git add -A`. Both are
  now gitignored, so the coverage check below confirms that rather than
  reporting a gap.
- Private keys / certs: `*.pem`, `*.key`, `*.pfx`, `*.p12`, `*.keytab`,
  `id_rsa`, `id_dsa`.
- Credential dumps: `credentials.json`, `service-account*.json`,
  `*.kdbx`, `.netrc`, `.pgpass`, `.htpasswd`.

A tracked secret file is **Must Fix**: `git rm --cached` it, add it to
`.gitignore`, and **rotate** anything it exposed (it is already in history).

### `.gitignore` coverage

Confirm `.env` and the patterns above are gitignored, not merely absent. A
secret that is untracked today but not ignored can be committed by
`git add -A`.

### Hardcoded secrets — detect-secrets + grep fallback

Primary: run `detect-secrets` against the baseline (see Steps). It catches
token shapes AND high-entropy strings the regexes below would miss, and the
baseline suppresses audited false positives.

Fallback / supplement (also useful for explaining a finding): scan for
assignments of a secret-looking name to a literal, and known token shapes.
Pattern reference (ripgrep regex):

| What | Pattern |
|---|---|
| Secret-named literal | `(?i)(pass(word|wd)?\|secret\|token\|api[_-]?key\|client[_-]?secret\|access[_-]?key\|auth[_-]?token\|private[_-]?key)\s*[:=]\s*["'][^"']{6,}["']` |
| Private key block | `-----BEGIN (RSA \|EC \|OPENSSH \|DSA \|PGP )?PRIVATE KEY-----` |
| AWS access key id | `AKIA[0-9A-Z]{16}` |
| GitHub token | `gh[pousr]_[A-Za-z0-9]{36,}` or `github_pat_[A-Za-z0-9_]{60,}` |
| Slack token | `xox[baprs]-[A-Za-z0-9-]{10,}` |
| Bearer/JWT | `(?i)bearer\s+[A-Za-z0-9._\-]{20,}` / `eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}` |
| URL with embedded creds | `[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@` |
| Connection-string password | `(?i)(password\|pwd)=[^;"'\s]{4,}` |
| Modal token | `a[ks]-[A-Za-z0-9]{20,}` |

For each hit, **redact the value in your report** — show the variable name
and first few characters only, never the full secret.

### Secrets passed to logging or print

ruff `S` does not cover this. Flag log and print statements that
interpolate a secret-bearing value: `Authorization` headers, anything named
`*secret*` / `*token*` / `*password*`, or a whole `headers` / `auth` dict.
Even at DEBUG a logged token ends up in the deployment's log store, which
is usually retained longer and read by more people than the code.

### Insecure patterns — ruff `S` rules

`ruff check --select S` reports these patterns; cite the rule code in each
finding (as `code-quality-review` cites ruff codes):

- **Hardcoded password** — `S105` (string), `S106` (func arg), `S107`
  (default arg).
- **`eval` / `exec`** — `S307` (eval), `S102` (exec).
- **Unsafe deserialization** — `S301` (`pickle`), `S506` (`yaml.load`
  without `SafeLoader`). Also `torch.load` and darts' `.load()`: flag any
  checkpoint load whose path could come from outside this project's own R2
  artifacts.
- **SQL injection** — `S608` (string-built query). DuckDB queries built by
  string are the case to judge, not to wave through: fixed table and column
  constants are a false positive, anything reaching a node name or a date
  from a feed is a finding.
- **Swallowed errors** — `S110` (try/except/pass), which hides the failure
  that would otherwise surface a credential problem at its source.
- **Shell injection surface** — `S602`/`S604`/`S605` (`shell=True`),
  `S603` (untrusted input), `S607` (partial executable path).
- **TLS verification disabled** — `S501` (`verify=False`).
- **Other** — `S104` (bind all interfaces), `S324` (weak hash).

Treat any `S`-rule hit on the target files as **at least Should Fix**;
`S105-S107` (hardcoded secret) is **Must Fix**. Do not re-flag the
configured ignores below.

## Configured exceptions (don't re-flag these)

`pyproject.toml` encodes the legitimate exceptions and `.secrets.baseline`
records audited non-secrets — respect both. **Read the actual per-file
ignores in `pyproject.toml` rather than assuming**; that file is the
authority and this section can fall behind it.

- **`tests/**` here ignores `S101` *and* `S105-S107`**, on the stated grounds
  that the test data is code-generated fixtures rather than real credentials.
  That is a real gap, not a reassurance: **ruff will not report a hardcoded
  credential under `tests/`**, so a clean `ruff --select S` run says nothing
  about test files. `detect-secrets` and the grep patterns are what cover
  `tests/`, and a credential found there is still a **Must Fix** with
  rotation. Do not cite a clean ruff run as evidence for `tests/`.
- **`S108` (hardcoded `/tmp`) fires 19 times under `tests/` and is known
  noise.** Twelve in `test_utils_s3.py` and seven in `test_modeling_load.py`,
  every one a `/tmp/...` string inside a mock assertion rather than a path
  the code writes to. `pyproject.toml` does not ignore `S108`, so these
  recur on every run; report a *new* `S108` outside those two files, not
  these.

## The `.env.example` caveat

- **`.env.example` is supposed to be committed.** Do NOT flag it as a
  tracked secret file. Instead, **open it and confirm every value is a
  placeholder** — a real value leaked into the template IS a finding, and a
  Must Fix.

## Repo-specific caveats

- **Values that look secret-shaped here and are not.** `AWS_S3_BUCKET`,
  `AWS_S3_FOLDER`, `S3_ENDPOINT_URL`, `AWS_DEFAULT_REGION` and `MAX_JOBS` are
  pointer configuration. Do not flag a hardcoded bucket name or endpoint URL;
  do flag a hardcoded access key or token. The R2 endpoint URL embeds the
  Cloudflare account id — low-sensitivity config, not a credential. Modal app
  and secret *names* (`aws-secret`, the app names under `modal_jobs/`) are
  identifiers too. Without this list every run re-flags the same five values.
- **`.env.example` is a tracked template and the intended exception**, but
  open it: every value must still be a placeholder or non-secret default
  (`AWS_DEFAULT_REGION=auto`, the bucket name, empty key fields). A real
  access key leaked into the template IS a finding, and a Must Fix.
- **SPP portal URLs are unauthenticated — keep it that way in logs.** The
  collection code logs the URLs it fetches; that is fine only while they carry
  no token. A signed or keyed URL reaching a log statement is a finding.
- **This repo has a real credential surface, unlike most.** The R2/S3 key
  pair, a Modal token, and the Posit Connect (rsconnect) API key used for
  deploys are all live secrets. They must come from `os.environ`,
  `python-dotenv`, or the Modal `aws-secret` — never a literal. A hardcoded
  one here is not a theoretical finding.
- **Credential files this repo can plausibly grow**, on top of the standard
  list: `.modal.toml` and `rsconnect-python/` config directories. Both hold
  working tokens once written, and both are exactly the kind of file that
  arrives with a `git add -A`.
- **`manifest.json`, `uv.lock` and `requirements.txt` are excluded by the
  committed baseline**, along with the baseline itself. The Posit Connect
  manifest records a per-file md5 for everything it ships and every one reads
  as high-entropy hex; they change on every deploy, so a recorded finding for
  them is stale by the next commit. The exclusion lives in the baseline's
  `filters_used` as a `should_exclude_file` pattern —
  `uv\.lock|requirements\.txt|\.secrets\.baseline|manifest\.json` — and
  any regeneration has to reproduce it. Do not report a hit in those files as
  a credential without opening it.
- **DuckDB queries built by string are the `S608` case to judge, not to
  wave through.** Where the interpolated part is a fixed table or column
  constant it is a false positive; where it reaches a node name, a date, or
  anything that arrived from a feed, it is a finding.
- **`notebooks/**` ignores `S101` along with the marimo lint ignores**, because
  a marimo cell uses `assert` as an interactive sanity check. The
  hardcoded-secret rules `S105-S107` stay **active** there — a literal token
  in a notebook cell is exactly as committed as one in `src/`.

Each caveat has to be anchored to something durable: a file that exists, a
documented convention. A caveat naming a deleted module authorizes an
exception that no longer applies.

## Output format

Group findings by severity, same buckets as `code-quality-review` so the
`code-reviewer` agent's report stays uniform. **Number findings
sequentially**, continuing from the review findings rather than restarting
at 1, so a closing list of open items is unambiguous about which section
each belongs to.

### Must Fix
- Confirmed live secret in code, a tracked `.env`/key file, a real value
  in `.env.example`, or an `S105-S107` hit. **Always include the
  remediation:** remove it, move it to an environment variable / `.env`,
  and **rotate the exposed credential** — state explicitly that
  working-tree removal is not enough if it was ever committed, since it
  persists in git history; scrub with `git filter-repo` / BFG if needed.

### Should Fix
- Other insecure-pattern `S`-rule hits (`verify=False`, `eval`/`exec`,
  unsafe deserialization, `shell=True`) and secret logging.

### Consider
- Possible-but-uncertain matches (entropy hits not in the baseline that
  might be fixtures), `.gitignore` gaps with nothing currently leaked.

For each finding: a sequential number, `file:line`, the rule code where
applicable, what matched (**redacted**), why it is a risk, and the
suggested fix.

## Steps

1. **Pick targets.** Path argument → that path. Otherwise
   `git diff --name-only HEAD` unioned with `git ls-files --others
   --exclude-standard`. Union, not fallback — see `code-quality-review`. The tracked-file and `.gitignore` checks run whole-repo
   regardless.
2. **Mechanical passes first** — reuse the configured tooling:
   - `uv run ruff check --select S <targets>` — insecure patterns. Cite
     each rule code.
   - `uv run detect-secrets-hook --baseline .secrets.baseline <targets>` —
     entropy/regex secret detection. The `-hook` entry point **reads** the
     baseline to suppress audited false positives, then exits non-zero and
     prints each new secret's type and location. Crucially it **does not
     rewrite `.secrets.baseline`** — unlike `detect-secrets scan
     --baseline .secrets.baseline`, which rewrites the file in place AND
     drops the recorded entries for every file outside `<targets>`, wiping
     audited false positives for everything unscanned. Treat every
     reported hit as a candidate and open it to confirm a real secret
     versus a placeholder or fixture.
   - **Check the baseline is there and intact before trusting a clean
     run.** It must exist, be tracked, and still carry its detector list.
     If it is missing or truncated, treat the scan as invalid even when the
     hook exits cleanly; a clean exit then does not prove the detector list
     or audited false positives were used.
   - **This skill is report-only: never modify `.secrets.baseline`.**
     Recording a new audited false positive (via `scan` plus an
     interactive `detect-secrets audit`) is a deliberate, separate human
     step — and **never** commit a real secret into the baseline to
     silence it. If the baseline shows as modified after a scan, restore
     it with `git checkout -- .secrets.baseline`.
   - **When you report a false positive, recommend the inline fix, not the
     baseline.** Marking the line `# pragma: allowlist secret` with the
     reason beside it clears the hit and keeps the reason beside the code.
     A baseline entry stores a hash without the reason it was allowed. So
     the recorded findings should be zero; the baseline should carry the
     detector list and the exclude pattern, not accumulated exceptions.
     Recommend excluding a whole file only where its values change on every
     run, since those are stale in a baseline by the next commit.
   - **Regenerating it must reproduce the exclude pattern**, which is not
     carried forward from the existing file — the flag below has to match
     what the committed baseline already excludes:

     ```bash
     uv run detect-secrets scan \
       --exclude-files 'uv\.lock|requirements\.txt|\.secrets\.baseline|manifest\.json' \
       > .secrets.baseline
     ```

     Dropping the flag brings back findings that change on every deploy,
     and a value that changes every run is stale in the baseline by the
     next commit — exclude the file instead. A Connect `manifest.json`,
     whose per-file md5s all read as high-entropy hex, is the case the
     pattern exists for. Commit the result: the pattern is stored *in* the
     baseline as a filter, so every later caller inherits it without
     repeating the flag.
3. **Tracked-file check.** `git ls-files`, filtered for the credential-file
   patterns above, applying the `.env.example` exception. Confirm
   `.gitignore` coverage.
4. **Fallback content scan.** Run the grep patterns over the target files
   for anything the tools did not surface, then check for secret logging.
   Open each hit to confirm it is a real secret versus a placeholder or
   fixture; redact before reporting.
5. **Emit the report** grouped by severity. Make **no edits** — this skill
   is report-only. If anything is Must Fix, lead with it. If there are no
   findings, say "No security findings" explicitly so the clean result is
   on record.

## Note on marimo notebooks

Scan notebook `.py` files for secrets the same as any source file — a
hardcoded token in a marimo cell is just as committed. The notebook ruff
ignores include `S101` only, so every other `S` rule — including the
hardcoded-secret rules `S105-S107` — still fires there. Read
`[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` rather than trusting
a list here; that file is the authority.
