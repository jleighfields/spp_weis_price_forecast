# `tests/`

Two suites with different scopes and speeds.

## `unit/` — fast, no network or browser

Pure `pytest` over `src/`. This is the default gate:

```bash
uv run pytest -m "not torch and not e2e" -q
```

Fixtures under `unit/fixtures/` are **trimmed real portal CSVs** — one
pre-launch (no `BAA` column, East-only) and one post-launch (both BAAs) per
feed — so the IM collector tests exercise the real schema, including the
blank-`BAA` rows the live feed produces.

## `e2e/` — Playwright driving the Shiny app

Browser tests of the rendered app. Run whenever `app.py` or `src/plotting.py`
changes:

```bash
uv run playwright install chromium   # one-time
uv run pytest -m e2e -q
```

`e2e/app_for_test.py` is a lightweight harness that imports the real `app.py`
UI/server and feeds it fixture data; `conftest.py` provides the Shiny app
fixture. `app_for_test.py` replaces `_do_load_data` and `_do_load_models`
with functions returning synthetic frames built in the file itself, so the
suite reaches no champion model, no checkpoint and no R2 bucket.
