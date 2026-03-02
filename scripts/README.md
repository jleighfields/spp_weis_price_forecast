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
