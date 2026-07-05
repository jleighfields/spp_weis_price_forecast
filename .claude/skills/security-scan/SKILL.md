---
name: security-scan
description: Scan for leaked secrets and insecure patterns — hardcoded passwords/tokens/keys, tracked .env or credential files, secrets in logs, and unsafe defaults. Drives ruff S + detect-secrets with a grep fallback; report-only, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [file-or-directory]
---

# Security Scan

Find secrets and unsafe patterns before they get committed, and report a
findings list. **Report-only** — never edit or delete a secret yourself.
Removing a secret and (critically) **rotating** it is a deliberate,
human-driven step.

This skill leans on tooling first, the same way `code-quality` leans on
ruff, then fills gaps with `git` checks and `grep`:

1. **`ruff` `S` rules (flake8-bandit)** — mechanical insecure-pattern
   detection (hardcoded passwords, `eval`/`exec`, `shell=True`, unsafe
   deserialization, weak hashes). Per-file exceptions are configured in
   `pyproject.toml`.
2. **`detect-secrets`** — entropy + regex secret detection with a committed
   `.secrets.baseline` so known false positives stay suppressed.
3. **`git ls-files` + `grep`** — tracked-credential-file checks and a regex
   fallback for token shapes the tools miss.

Its lens (does this leak a credential or open a hole?) is distinct from
`code-quality` (is it clear?) and `simplify-audit` (should it exist?).

## Arguments

- **file-or-directory** (optional): Path to scan for secret *content* /
  insecure patterns. If omitted, scan the changed files
  (`git diff --name-only HEAD`, plus untracked from `git status --short`).
  The tracked-file checks always run against the whole repo regardless of
  the argument.

## Scope

- **Content / pattern scan:** the target files (changed files by default,
  or the given path).
- **Tracked-file checks:** always whole-repo via `git ls-files` — a
  committed `.env` is a repo-wide fact, not a diff fact.
- **Never scan** gitignored/untracked artifacts for content (`.venv/`,
  `data/`, `models/`, `__marimo__/`, `__pycache__/`) — but DO still
  confirm they are gitignored.

## What to check

### 1. Tracked credential files (whole repo, always)

Run `git ls-files` and flag any tracked file that should never be committed:

- `.env`, `.env.local`, `.env.*` **except** `.env.example` / `.env.template`
  / `.env.sample` (those are intended templates — see caveat below).
- Private keys / certs: `*.pem`, `*.key`, `*.pfx`, `*.p12`, `*.keytab`,
  `id_rsa`, `id_dsa`.
- Credential dumps: `credentials.json`, `service-account*.json`,
  `*.kdbx`, `.netrc`, `.pgpass`, `.htpasswd`, `.modal.toml`,
  `rsconnect-python/` config directories.

A tracked secret file is **Must Fix**: `git rm --cached` it, add it to
`.gitignore`, and **rotate** anything it exposed (it is already in history).

### 2. `.gitignore` coverage

Confirm `.env` (and the patterns above) are gitignored, not merely
absent. A secret that is untracked today but not ignored is one
`git add -A` away from being committed. This repo's `.gitignore` already
covers `.env`.

### 3. Hardcoded secrets (target files) — detect-secrets + grep fallback

Primary: run `detect-secrets` against the baseline (see Steps). It catches
token shapes AND high-entropy strings the regexes below would miss, and the
baseline keeps audited false positives quiet.

Fallback / supplement (also useful for explaining a finding): scan for
assignments of a secret-looking name to a literal, and known token shapes.
Pattern reference (ripgrep regex):

| What | Pattern |
|---|---|
| Secret-named literal | `(?i)(pass(word|wd)?\|secret\|token\|api[_-]?key\|client[_-]?secret\|access[_-]?key\|auth[_-]?token\|private[_-]?key)\s*[:=]\s*["'][^"']{6,}["']` |
| Private key block | `-----BEGIN (RSA \|EC \|OPENSSH \|DSA \|PGP )?PRIVATE KEY-----` |
| AWS access key id | `AKIA[0-9A-Z]{16}` |
| GitHub token | `gh[pousr]_[A-Za-z0-9]{36,}` or `github_pat_[A-Za-z0-9_]{60,}` |
| Modal token | `a[ks]-[A-Za-z0-9]{20,}` |
| Bearer/JWT | `(?i)bearer\s+[A-Za-z0-9._\-]{20,}` / `eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}` |
| URL with embedded creds | `[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@` |
| Connection-string password | `(?i)(password\|pwd)=[^;"'\s]{4,}` |

This repo reads real credentials from the environment — the R2/S3 pair
**`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`** (provided locally via
the gitignored `.env` and in Modal via the `aws-secret` secret), plus
Modal tokens and the Posit Connect (rsconnect) API key used for deploys.
Those values must come from `os.environ` / `python-dotenv` / the Modal
secret, never a literal in source. A hardcoded value for any of them is a
**Must Fix**.

For each hit, **redact the value in your report** (show the variable name
and first few chars only, never the full secret).

### 4. Secrets passed to logging or print

ruff `S` does not cover this. This repo logs via the `logging` module —
flag log/print statements that interpolate a secret-bearing value
(`Authorization` headers, anything named `*secret*`/`*token*`/`*api*key*`,
or a whole `headers`/`auth` dict). The data-collection module logs the
URLs it queries; confirm those URLs don't carry an embedded API key in
the query string (SPP portal URLs are unauthenticated today — keep it
that way in logs).

### 5. Insecure patterns — ruff `S` rules

These fire mechanically from `ruff check --select S`; cite the rule code in
each finding (like `code-quality` cites ruff codes):

- **Hardcoded password** — `S105` (string), `S106` (func arg), `S107`
  (default arg).
- **`eval` / `exec`** — `S307` (eval), `S102` (exec).
- **Unsafe deserialization** — `S301` (`pickle`), `S506` (`yaml.load`
  without `SafeLoader`). Note: the model checkpoints are loaded with
  `torch.load` / darts `.load()` — flag any load of a checkpoint that
  comes from an untrusted path, but the project's own R2 artifacts are
  trusted.
- **Shell injection surface** — `S602`/`S604`/`S605` (`shell=True`),
  `S607` (partial executable path), `S603` (subprocess untrusted input).
- **TLS verification disabled** — `S501` (`verify=False`).
- **SQL injection** — `S608` (string-built query). DuckDB queries that
  interpolate a code-controlled table/path constant are fine; anything
  interpolating user/app input is a finding.
- **Other** — `S104` (bind all interfaces), `S324` (weak hash),
  `S110` (try/except/pass).

Treat any `S`-rule hit on the target files as **at least Should Fix**;
`S105-S107` (hardcoded secret) is **Must Fix**. Do not re-flag the
configured ignores (see "Configured exceptions" below).

## Configured exceptions (don't re-flag these)

`pyproject.toml` (`[tool.ruff.lint.per-file-ignores]`) and
`.secrets.baseline` encode the legitimate exceptions — respect both:

- **`tests/**`** ignores `S101`/`S105-S107`: pytest's whole model is
  `assert`, and test data is code-generated fixtures, not real creds.
- **`notebooks/**`** ignores `S101` (interactive `assert` sanity checks)
  along with the marimo lint ignores. The hardcoded-secret rules
  (`S105-S107`) stay **active** in notebooks — a literal token in a
  marimo cell is just as committed.
- **`manifest.json`, `uv.lock`, and `requirements.txt`** are excluded
  from the `detect-secrets` baseline: the Posit Connect manifest
  `"checksum"` hashes and lockfile hashes are high-entropy but are not
  secrets.

## False-positive caveats specific to this repo

Apply these so the scan is accurate, not noisy:

- **`.env.example` is a tracked template** (the real `.env` stays
  gitignored). It is the intended exception to the tracked-`.env.*` check —
  but open it and confirm every value is still a placeholder or non-secret
  default (`AWS_DEFAULT_REGION=auto`, the bucket name, empty keys); a real
  access key or endpoint credential leaked into the template IS a finding.
- **`AWS_S3_BUCKET` / `AWS_S3_FOLDER` / `S3_ENDPOINT_URL` /
  `AWS_DEFAULT_REGION` / `MAX_JOBS`** are pointer/config env vars, not
  secrets. Don't flag a hardcoded bucket name or endpoint URL; do flag a
  hardcoded access key or token. (The R2 endpoint URL does embed the
  Cloudflare account id — treat that as low-sensitivity config, not a
  credential.)
- **Modal app/secret *names*** (`aws-secret`, app names in
  `modal_jobs/`) are identifiers, not credentials.

## Output Format

Group findings by severity, same buckets as `code-quality` so the
code-reviewer agent's report stays uniform:

### Must Fix
- Confirmed live secret in code, a tracked `.env`/key file, or an
  `S105-S107` hit. **Always include the remediation:** remove it, move it to
  an environment variable / `.env`, and **rotate the exposed credential**
  (state explicitly that working-tree removal is not enough if it was ever
  committed — it persists in git history; scrub with `git filter-repo` / BFG
  if needed).

### Should Fix
- Other insecure-pattern `S`-rule hits (`verify=False`, `eval`/`exec`,
  unsafe deserialization, `shell=True`, SQL injection) and secret logging.

### Consider
- Possible-but-uncertain matches (entropy hits not in the baseline that
  might be fixtures), `.gitignore` gaps with nothing currently leaked.

For each finding: `file:line`, the rule code where applicable, what matched
(**redacted**), why it is a risk, and the suggested fix.

## Steps

1. **Pick targets.** Path argument → that path. Otherwise
   `git diff --name-only HEAD`; if empty, `git status --short` for
   untracked files. The tracked-file checks (1–2) run whole-repo regardless.
2. **Mechanical passes first** — reuse the configured tooling:
   - `uv run ruff check --select S <targets>` — insecure patterns (check 5).
     Cite each rule code.
   - `uv run detect-secrets-hook --baseline .secrets.baseline <targets>` —
     entropy/regex secret detection (check 3). The `-hook` entry point
     **reads** the baseline to suppress audited false positives, then exits
     non-zero and prints each new secret's type + location. Crucially it
     **does not rewrite `.secrets.baseline`** — unlike
     `detect-secrets scan --baseline .secrets.baseline`, which rewrites the
     file in place AND drops the recorded entries for every file outside
     `<targets>`. Treat every reported hit as a candidate and open it to
     confirm a real secret vs a placeholder/fixture.
   - **This skill is report-only: never modify `.secrets.baseline`.** Do not
     run `detect-secrets scan --baseline .secrets.baseline` here. Recording a
     new audited false positive into the baseline (via `scan` + an
     interactive `detect-secrets audit .secrets.baseline`) is a deliberate,
     separate human step — and **never** commit a real secret into the
     baseline to silence it. If `.secrets.baseline` shows as modified after a
     scan, restore it with `git checkout -- .secrets.baseline`.
3. **Tracked-file check.** `git ls-files`, filter for the secret-file
   patterns in check 1, applying the `.env.example` exception. Confirm
   `.gitignore` coverage (check 2).
4. **Fallback content scan.** Run the check-3 grep patterns over the target
   files for anything the tools did not surface, then check 4 (secret
   logging). Open each hit to confirm it is a real secret vs a
   placeholder/fixture; redact before reporting.
5. **Emit the report** grouped by severity. Make **no edits** — this skill
   is report-only. If anything is Must Fix, lead with it.

## Note on marimo notebooks

Scan notebook `.py` files for secrets the same as any source file — a
hardcoded token in a marimo cell is just as committed. The notebook ruff
ignores include `S101` (interactive `assert` sanity checks) but NOT the
hardcoded-secret rules, so a literal token in a notebook cell still
surfaces.
