# R2 bucket reorg: `spp-weis-forecast` → `spp-rto`

Rename the R2 bucket and reorganize its top-level layout to drop the WEIS
branding and separate market data from model artifacts. This is a **live**
system (the IM collection Modal job writes every 4 h; the Shiny app reads on
every load; the retrain job writes weekly), so the copy and the cutover are
sequenced to avoid data loss.

## Target layout

Current (`spp-weis-forecast`, 96,137 objects, 3.73 GB):

```
spp-weis-forecast/
  data/            59,308 obj  696 MB   WEIS history (frozen, archival)
  data_im/         36,726 obj  433 MB   IM market data (LIVE — app reads this)
  model_retrains/     102 obj  2.6 GB   model checkpoints
  S3_models/            1 obj    ~0     champion.json (pointer)
```

New (`spp-rto`):

```
spp-rto/
  im/              ← data_im/
  weis/            ← data/
  models/
    retrains/<ts>/ ← model_retrains/<ts>/
    champion.json  ← S3_models/champion.json   (pointer rewritten)
```

`AWS_S3_FOLDER` stays `""` (data at bucket root). The only **live read path**
is `data_im/` → `im/` and the model prefixes; `weis/` is archival (no live
reader — only the unused `weis_stitch_fill.py`).

## Where the prefixes live in code (what changes)

| Prefix | Definition site(s) | New value |
|--------|--------------------|-----------|
| `data_im/` (IM data) | `src/data_collection_im.py::get_s3_base_path_im` (L93); **duplicated** in `src/data_engineering.py:133`; `scripts/weis_stitch_fill.py:72` | `im/` |
| `data/` (WEIS data) | `src/data_collection.py::get_s3_base_path` (L79) + hardcoded L769/771/781; `scripts/weis_stitch_fill.py:72` | `weis/` |
| `model_retrains/` | `src/utils.py:27 RETRAINS_PREFIX` | `models/retrains/` |
| `S3_models/champion.json` | `src/utils.py:28 CHAMPION_KEY_SUFFIX`; `get_loaded_models` default (L62) | `models/champion.json` |

`scripts/r2_promote_champion.py` and `notebooks/model_training/model_retrain.py`
import `RETRAINS_PREFIX` / `CHAMPION_KEY_SUFFIX` from `utils`, so they follow
the constants automatically — only their comments need a sweep.

## Phase 1 — Code changes (feature branch, NOT deployed until cutover)

1. **`src/utils.py`** — `RETRAINS_PREFIX = "models/retrains/"`,
   `CHAMPION_KEY_SUFFIX = "models/champion.json"`, align the `get_loaded_models`
   default off the new prefix; update docstrings/comments.
2. **Collapse the IM-prefix duplicate (single source of truth).** Introduce
   `IM_PREFIX = "im/"` and `WEIS_PREFIX = "weis/"` in `src/data_collection_utils.py`
   (already the neutral storage-helper home, import-safe for both collectors and
   `data_engineering.py`). Have `get_s3_base_path_im`, `get_s3_base_path`, and
   `data_engineering.py:133` all read those constants instead of hardcoding the
   literal — kills the `data_engineering.py:133` duplicate.
3. **`src/data_collection.py`** (WEIS, dead but keep correct) — `get_s3_base_path`
   + L769/771/781 use `WEIS_PREFIX`.
4. **`scripts/weis_stitch_fill.py:72`** — `weis/`, `im/`.
5. **Comment sweep** — `data_im/` mentions in `data_collection_im.py`,
   `data_engineering.py`, the IM backfill notebook; `S3_models/`/`model_retrains/`
   mentions in `utils.py`, `r2_promote_champion.py`, `model_retrain.py`.
6. **Modal bucket name → code `env=` (Option B).** The bucket name is not
   secret. Set `env={"AWS_S3_BUCKET": "spp-rto"}` on each Modal function in
   `modal_jobs/data_collection_im.py` and `modal_jobs/model_retrain.py`
   (alongside the existing `MAX_JOBS`), so the Modal cutover is a pure
   `modal deploy` — no secret edit. `aws-secret` is trimmed **once** to drop its
   `AWS_S3_BUCKET` key (so a stale value there can't shadow the code value —
   Modal's secret-vs-`env=` precedence is not worth relying on); the secret keeps
   the credentials, `S3_ENDPOINT_URL`, `AWS_DEFAULT_REGION`, and `AWS_S3_FOLDER`.
   Do **not** move `S3_ENDPOINT_URL` into code — it embeds the R2 account ID and
   stays out of git. Update the CLAUDE.md convention line: the bucket name lives
   in the Modal `env=` (code); credentials/endpoint/region/folder stay in
   `aws-secret`.
7. **Tests** — `tests/unit/test_utils_s3.py` (`S3_models/`→`models/`,
   `model_retrains/ts/`→`models/retrains/ts/`, champion-key assertions). Add any
   needed coverage for the new `IM_PREFIX`/`WEIS_PREFIX` constants. Run
   `uv run pytest tests/unit -q`.
8. **`.env.example`** — `AWS_S3_BUCKET=spp-rto`.

## Phase 2 — R2 copy (server-side, key-remapped) — ✅ DONE

Implemented as `scripts/r2_reorg_copy.py` (dry-run by default; `--execute` to
copy). Bulk copy verified: `spp-rto` holds 96,137 objects — `im/` 36,726,
`weis/` 59,308, `models/` 103 (102 checkpoints + `models/champion.json`, both
its pointers rewritten), byte sizes matching source, and zero old-prefix keys.
The script is idempotent (skip-existing), so re-running it is the Phase-3 delta
sync. A re-run already showed the live IM job had added ~293 objects to the old
bucket since the bulk copy — the reason the cutover needs a final sync.

The mechanics below describe that script:

One script (`scripts/` one-off, run locally with `.env` creds), threaded
`copy_object` (server-side, no download; every object is < 5 GB so single-part
copy works):

- Create `spp-rto` (idempotent — tolerate `BucketAlreadyOwnedByYou`).
- Copy with key remap: `data/*`→`weis/*`, `data_im/*`→`im/*`,
  `model_retrains/*`→`models/retrains/*`.
- `S3_models/champion.json` → `models/champion.json`, **rewriting**
  `champion_artifact_folder`: `model_retrains/<ts>/` → `models/retrains/<ts>/`.
- Skip keys already present in the target (resumable / re-runnable).
- Verify: per-prefix object counts + total bytes match source→target mapping.

This bulk copy is safe to run anytime before cutover; the old bucket keeps
serving live traffic throughout.

## Phase 3 — Cutover — ✅ DONE

Executed and verified:
- Final delta sync (96,430→96,430, missing=0).
- Both Modal apps redeployed (v2). `aws-secret` trimmed to drop `AWS_S3_BUCKET`
  (credentials/endpoint/region/folder kept), so the code `env=spp-rto` is the
  sole bucket source. A manual invocation of the **deployed** `collect_im_hourly`
  wrote fresh LMP to `spp-rto/im/` — Option B confirmed.
- App read path resolves the champion from `spp-rto/models/` + reads `im/`
  tables. Posit Connect redeployed on the new code; forecast renders.

Watch-items before Phase 4:
- **Next scheduled IM collection** (every 4 h) lands in `spp-rto/im/`.
- **First scheduled retrain** (Sun 20:00 UTC, next 2026-07-12) writes to
  `spp-rto/models/retrains/<ts>/` and updates `spp-rto/models/champion.json`.

Original sequence, for reference:



Sequence to avoid losing writes (IM job every 4 h; retrain Sun 20:00 UTC —
avoid both):

1. **Final delta sync** — re-run the copy script (skip-existing makes it a
   fast delta) to catch any `data_im/`/model writes since the bulk copy.
2. **Flip config + redeploy together:**
   - Local `.env`: `AWS_S3_BUCKET=spp-rto`.
   - **Trim `aws-secret`** to drop its `AWS_S3_BUCKET` key (keep credentials,
     `S3_ENDPOINT_URL`, `AWS_DEFAULT_REGION`, `AWS_S3_FOLDER`) — *one-time user
     action* (Modal dashboard or `modal secret create`); I can't edit Modal
     secrets. After this, Modal never needs a secret edit for a bucket change.
   - `modal deploy modal_jobs/data_collection_im.py` and
     `modal deploy modal_jobs/model_retrain.py` — bakes Phase-1 code **and** the
     new `env=` bucket, so this is the whole Modal cutover.
   - **Posit Connect** app env var `AWS_S3_BUCKET=spp-rto` — *user action* (the
     app reads `os.environ`; Posit env is dashboard-set), then redeploy the app.
3. **Verify:** app loads champion from `models/champion.json` +
   `models/retrains/<ts>/`; app reads `im/`; `modal run
   modal_jobs/data_collection_im.py::collect_im_hourly` writes to `im/`;
   confirm the next scheduled collection + retrain land correctly.

**Rollback:** revert the env var (+ redeploy old code). The old bucket is
untouched by the copy, so it remains a complete, current fallback until the
final sync; keep the flip window free of collection/retrain runs.

## Phase 4 — Decommission

- Keep `spp-weis-forecast` as a backup ~2 weeks.
- After the new bucket is validated across a full collection + retrain cycle,
  delete the old bucket (or empty it) to stop paying for the duplicate 3.7 GB.

## Open / notes

- The one-time `aws-secret` trim + the Posit env change are the only steps I
  can't do programmatically — they need you (or a `!`-run command in session).
  Everything else (code, the R2 copy, `modal deploy`) I can do.
- The `weis/` copy is purely archival; if the stitch path is truly never
  coming back we could drop `weis/` later, but copying it now is cheap
  insurance and was the chosen option.
