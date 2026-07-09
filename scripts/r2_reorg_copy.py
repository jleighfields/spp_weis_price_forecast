"""One-time R2 reorg copy: spp-weis-forecast -> spp-rto.

Server-side clones every object from the source bucket into the target bucket
under the new top-level layout, remapping keys:

    data/*            -> weis/*
    data_im/*         -> im/*
    model_retrains/*  -> models/retrains/*
    S3_models/champion.json -> models/champion.json   (pointer rewritten)

`champion.json` is special-cased: its `champion_artifact_folder` (and any
other value pointing at the old model_retrains/ layout) is rewritten to the
new prefix before upload, so the promoted champion still resolves.

Copies are server-side (`copy_object`, no download) and skip keys already
present in the target, so the script is idempotent and safe to re-run as a
delta sync right before cutover. Credentials/endpoint come from `.env`.

Usage:
    python scripts/r2_reorg_copy.py            # dry run: show the plan, touch nothing
    python scripts/r2_reorg_copy.py --execute  # create spp-rto and copy
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("r2_reorg_copy")

SRC_BUCKET = "spp-weis-forecast"
TGT_BUCKET = "spp-rto"
CHAMPION_SRC_KEY = "S3_models/champion.json"
CHAMPION_TGT_KEY = "models/champion.json"
MAX_WORKERS = 32

# Ordered longest/most-specific first; first match wins.
PREFIX_MAP = [
    ("data_im/", "im/"),
    ("data/", "weis/"),
    ("model_retrains/", "models/retrains/"),
    ("S3_models/", "models/"),
]


def remap_key(key: str) -> str:
    """Map a source object key to its target key under the new layout."""
    for old, new in PREFIX_MAP:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def _client():
    # Size the connection pool to the worker count so concurrent copies don't
    # thrash the default 10-connection pool.
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        config=Config(max_pool_connections=MAX_WORKERS),
    )


def _key_exists(s3, bucket: str, key: str) -> bool:
    """True if the object exists (False on 404)."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def list_keys(s3, bucket: str) -> list[str]:
    """Return all object keys in a bucket (empty list if the bucket is absent)."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket):
            keys.extend(o["Key"] for o in page.get("Contents", []))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchBucket", "404"):
            return []
        raise
    return keys


def ensure_bucket(s3, bucket: str, execute: bool) -> None:
    """Create the target bucket if it does not already exist.

    Args:
        s3: Boto3 S3 client.
        bucket: Target bucket name.
        execute: If False, only log what would happen (dry run).
    """
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket in existing:
        log.info(f"target bucket {bucket!r} already exists")
        return
    if not execute:
        log.info(f"[dry-run] would create bucket {bucket!r}")
        return
    s3.create_bucket(Bucket=bucket)
    log.info(f"created bucket {bucket!r}")


def copy_champion(s3, execute: bool) -> None:
    """Copy champion.json with its pointer values rewritten to the new layout.

    Skips if the target champion already exists, so re-running the script as a
    delta sync never reverts a champion that was promoted directly into the
    target bucket after cutover.
    """
    if _key_exists(s3, TGT_BUCKET, CHAMPION_TGT_KEY):
        log.info(
            f"{TGT_BUCKET}/{CHAMPION_TGT_KEY} already exists; leaving it untouched"
        )
        return
    try:
        body = s3.get_object(Bucket=SRC_BUCKET, Key=CHAMPION_SRC_KEY)["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            log.info("no champion.json in source; skipping")
            return
        raise
    cfg = json.loads(body)
    rewritten = {k: (remap_key(v) if isinstance(v, str) else v) for k, v in cfg.items()}
    changed = {k: (cfg[k], rewritten[k]) for k in cfg if cfg[k] != rewritten[k]}
    log.info(f"champion.json pointer rewrites: {changed or 'none'}")
    if not execute:
        log.info(f"[dry-run] would write {TGT_BUCKET}/{CHAMPION_TGT_KEY}")
        return
    s3.put_object(
        Bucket=TGT_BUCKET,
        Key=CHAMPION_TGT_KEY,
        Body=json.dumps(rewritten, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log.info(f"wrote {TGT_BUCKET}/{CHAMPION_TGT_KEY}")


def main() -> int:
    """Plan and (optionally) run the reorg copy, returning a process exit code.

    Returns:
        0 on success (or dry run), 1 if any copy failed or a source object is
        missing from the target after the copy.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--execute",
        action="store_true",
        help="create the bucket and copy (default: dry run)",
    )
    args = ap.parse_args()
    execute = args.execute

    # The prefix remap operates on bare object keys, and champion.json's
    # pointer values are AWS_S3_FOLDER + folder — both assume an empty folder.
    # A non-empty folder would silently mis-map, so fail loudly instead.
    folder = os.environ.get("AWS_S3_FOLDER", "")
    assert folder == "", f"AWS_S3_FOLDER must be '' for this reorg; got {folder!r}"

    s3 = _client()
    ensure_bucket(s3, TGT_BUCKET, execute)

    src_keys = list_keys(s3, SRC_BUCKET)
    tgt_existing = set(list_keys(s3, TGT_BUCKET))
    log.info(
        f"source objects: {len(src_keys):,}  |  already in target: {len(tgt_existing):,}"
    )

    # champion.json is handled separately (content rewrite), not a plain copy.
    plain = [k for k in src_keys if k != CHAMPION_SRC_KEY]

    # Plan: source key -> target key, skipping ones already copied.
    todo = []
    by_prefix = defaultdict(int)
    for k in plain:
        tk = remap_key(k)
        by_prefix[tk.split("/", 1)[0] + "/"] += 1
        if tk not in tgt_existing:
            todo.append((k, tk))

    log.info(
        "target distribution (all objects): "
        + ", ".join(f"{p}={n:,}" for p, n in sorted(by_prefix.items()))
    )
    log.info(
        f"to copy this run: {len(todo):,} (skipping {len(plain) - len(todo):,} already present)"
    )

    copy_champion(s3, execute)

    if not execute:
        log.info(
            "[dry-run] no objects copied. Re-run with --execute to perform the copy."
        )
        return 0

    def _copy(pair):
        src_key, tgt_key = pair
        s3.copy_object(
            Bucket=TGT_BUCKET,
            Key=tgt_key,
            CopySource={"Bucket": SRC_BUCKET, "Key": src_key},
        )
        return tgt_key

    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_copy, p): p for p in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                errors.append((futures[fut][0], str(e)))
            done += 1
            if done % 5000 == 0 or done == len(todo):
                log.info(f"  copied {done:,}/{len(todo):,}")

    if errors:
        log.error(f"{len(errors)} copy errors; first few: {errors[:5]}")
        return 1

    # Verify. Build the set of target keys every source object should map to
    # (plain keys remapped, plus the champion). A collision in the prefix map
    # would shrink this set below the source count, so assert the mapping is
    # 1:1 before checking presence — a presence-only check could false-pass.
    expected = {remap_key(k) for k in plain}
    expected.add(CHAMPION_TGT_KEY)
    if len(expected) != len(plain) + 1:
        log.error(
            f"prefix map is not 1:1: {len(plain) + 1} source keys collapsed to "
            f"{len(expected)} target keys — aborting before trusting the copy."
        )
        return 1

    tgt_after = set(list_keys(s3, TGT_BUCKET))
    missing = sorted(expected - tgt_after)
    log.info(
        f"verify: source={len(src_keys):,} target={len(tgt_after):,} missing={len(missing):,}"
    )
    if missing:
        log.error(f"missing after copy (first few): {missing[:5]}")
        return 1
    log.info("DONE — all source objects present in target under the new layout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
