# Scripts

Utility scripts for managing the SPP WEIS price forecast infrastructure.

## r2_move_objects.py

Move, copy, or delete objects within a Cloudflare R2 bucket by key prefix.

- **Batch deletes**: uses the `delete_objects` API (1000 keys per request) instead of deleting one at a time
- **Concurrent copies**: uses a thread pool (`cpu_count - 1` threads) since there is no bulk copy API
- **Dry run by default**: always preview changes before committing

### Prerequisites

Requires these environment variables (set in `.env` or shell):

- `AWS_S3_BUCKET` - R2 bucket name (default bucket, can be overridden with `--bucket`)
- `S3_ENDPOINT_URL` - R2 endpoint URL
- `AWS_ACCESS_KEY_ID` - R2 access key
- `AWS_SECRET_ACCESS_KEY` - R2 secret key
- `AWS_DEFAULT_REGION` - set to `auto` for R2

### Usage

```bash
# 1. Dry run - preview what would move (no changes made)
uv run python scripts/r2_move_objects.py "old/prefix/" "new/prefix/"

# 2. Copy objects to new prefix (originals kept)
uv run python scripts/r2_move_objects.py "old/prefix/" "new/prefix/" --copy

# 3. Copy and delete originals
uv run python scripts/r2_move_objects.py "old/prefix/" "new/prefix/" --copy --delete

# 4. Delete all objects under a prefix (no copy)
uv run python scripts/r2_move_objects.py "old/prefix/" --delete-only

# 5. Operate on a different bucket (overrides AWS_S3_BUCKET)
uv run python scripts/r2_move_objects.py "" --bucket other-bucket --delete-only
```

### Flags

| Flag | Description |
|------|-------------|
| `--copy` | Perform the copy (without this, it's a dry run) |
| `--delete` | Delete originals after copy (requires `--copy`) |
| `--delete-only` | Delete objects without copying (cannot combine with `--copy`) |
| `--bucket` | Override the bucket name (default: `AWS_S3_BUCKET` env var) |

### Examples

#### Flatten a folder structure

```bash
# Step 1: dry run to verify mappings
uv run python scripts/r2_move_objects.py \
  "unity-catalog/7474645306723306/spp-weis/data/" "data/"

# Step 2: copy to new prefix
uv run python scripts/r2_move_objects.py \
  "unity-catalog/7474645306723306/spp-weis/data/" "data/" --copy

# Step 3: verify new paths work, then delete originals
uv run python scripts/r2_move_objects.py \
  "unity-catalog/7474645306723306/spp-weis/data/" --delete-only
```

#### Empty and delete a bucket

```bash
# Step 1: preview contents
uv run python scripts/r2_move_objects.py "" --bucket old-bucket

# Step 2: delete all objects
uv run python scripts/r2_move_objects.py "" --bucket old-bucket --delete-only

# Step 3: delete the empty bucket from Cloudflare dashboard
```

#### Move objects between prefixes (copy + delete in one step)

```bash
uv run python scripts/r2_move_objects.py "old/models/" "new/models/" --copy --delete
```

## r2_reorg_copy.py

One-time migration that clones the whole `spp-weis-forecast` bucket into
`spp-rto` under the current top-level layout, remapping keys
(`data/`→`weis/`, `data_im/`→`im/`, `model_retrains/`→`models/retrains/`,
`S3_models/champion.json`→`models/champion.json`). Unlike the general
`r2_move_objects.py`, it applies the full multi-prefix map in one pass and
rewrites `champion.json`'s pointers to the new model layout.

- **Server-side `copy_object`**: no download; threaded (`MAX_WORKERS`)
- **Idempotent**: skips keys already in the target, so a re-run is a delta
  sync (used to catch writes made between the bulk copy and cutover)
- **Champion-safe**: leaves an existing target `models/champion.json`
  untouched, and asserts `AWS_S3_FOLDER == ""` (the layout the remap assumes)
- **Verifies**: asserts the prefix map is 1:1, then that every source object
  is present in the target

```bash
uv run python scripts/r2_reorg_copy.py            # dry run
uv run python scripts/r2_reorg_copy.py --execute  # create spp-rto and copy
```

## r2_promote_champion.py

Promote (or revert to) a retrained model by repointing a forecast target's
`models/<target>/champion.json` — the pointer the Shiny app reads to decide
which `models/<target>/retrains/<timestamp>/` folder to serve. Each target
(`parameters.TARGETS`: `da` day-ahead, `rt` real-time) has its own champion;
`--target` selects which one, defaulting to `da` (the primary model).

Retrains that run with `PROMOTE_CHAMPION=false` are *staged*: their
checkpoints land in a timestamped folder but champion.json is untouched, so
the live app keeps its current model. This script is the manual promote/revert
step — it only rewrites the small JSON pointer (matching the schema
`model_retrain.py` writes), never moving checkpoints.

- **Target-scoped**: operates on one target's namespace at a time via
  `--target` (default `da`); pass `--target rt` for the real-time model.
- **Dry run by default**: prints the current → target change; pass `--promote`
  to write.
- **Validated**: refuses to point at a folder with no objects, so a typo can't
  break model loading.

### Prerequisites

Same R2 environment variables as `r2_move_objects.py` above.

### Usage

```bash
# List the retrain folders available to promote (newest last)
uv run python scripts/r2_promote_champion.py --list

# Show the model champion.json currently points at
uv run python scripts/r2_promote_champion.py --show

# Dry run - preview the change (no write)
uv run python scripts/r2_promote_champion.py 2026-07-06_12-41-45

# Promote / revert to that folder
uv run python scripts/r2_promote_champion.py 2026-07-06_12-41-45 --promote

# Operate on the real-time target instead of the default day-ahead one
uv run python scripts/r2_promote_champion.py --target rt --list
```

## tune_parameters.py

Bakes the top-N Optuna trials for a target into `parameters.TIDE_PARAMS_<TARGET>`
— the deterministic "update the params" step of a parameter sweep (no
hand-editing of param dicts). Reads the target's TiDE study from the Optuna
sqlite DB, takes the top-N complete trials by CRPS, and rewrites the marked
`# >>> TIDE_PARAMS_<TARGET> >>>` block in `src/parameters.py`.

- **Target-scoped**: `--target` (default `da`); only that target's block is touched.
- **Dry run by default**: prints the top trials + the new block; `--write` applies.
- Used standalone or as the params step of the `/tune-parameters` skill.

```bash
# Preview the top-5 DA trials and the block that would be written
uv run python scripts/tune_parameters.py --target da

# Apply (rewrite TIDE_PARAMS_DA in parameters.py)
uv run python scripts/tune_parameters.py --target da --write
```

## weis_stitch_fill.py

One-time WEIS→`im/` West stitch-fill (Phase 2 of the RTO West
migration). Copies the pre-launch WEIS (`weis/`) consolidated
lmp/mtlf/mtrf West rows into the `im/` tables with `BAA='SWPW'` and
`source='weis'`, giving the West BAA a continuous training series across
the 2026-04-01 seam.

- **LMP**: exact-name West nodes (`node_list.WEST_HUB_BA_NODES`) are
  copied straight through; the flagship `SWPW_HUB` has no WEIS equivalent,
  so its pre-launch history is proxied by the per-interval mean of all
  WEIS `WACM*` nodes.
- **MTLF/MTRF**: WEIS system-wide forecasts become the West BAA series.
- **No stitch** for da_lmp / rf_reserve_zone — WEIS had neither, so West
  values for those start at RTO launch.
- **Idempotent**: re-running drops the prior `source='weis'` rows before
  re-merging, and fails loud on any duplicate upsert key.

### Prerequisites

Same R2 environment variables as `r2_move_objects.py` above.

### Usage

```bash
# Preview row counts without writing
uv run python scripts/weis_stitch_fill.py --dry-run

# Materialize the stitch into the im/ tables
uv run python scripts/weis_stitch_fill.py
```
