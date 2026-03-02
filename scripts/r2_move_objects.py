"""Move or delete objects within a Cloudflare R2 bucket by key prefix.

Uses batch deletes (1000 per request) and concurrent copies (thread pool)
for faster operations on large numbers of objects.

Usage:
    python scripts/r2_move_objects.py <old_prefix> <new_prefix> [--copy] [--delete] [--delete-only]

Examples:
    # Dry run (default) - shows what would be moved
    python scripts/r2_move_objects.py "old/prefix/" "new/prefix/"

    # Copy objects to new prefix
    python scripts/r2_move_objects.py "old/prefix/" "new/prefix/" --copy

    # Copy and delete originals
    python scripts/r2_move_objects.py "old/prefix/" "new/prefix/" --copy --delete

    # Delete objects under a prefix (no copy)
    python scripts/r2_move_objects.py "old/prefix/" --delete-only

    # Delete all objects in a different bucket
    python scripts/r2_move_objects.py "" --bucket other-bucket --delete-only
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from dotenv import load_dotenv

# Load .env with override so it takes precedence over any existing env vars
load_dotenv(override=True)

# Reserve one core for the OS / other processes
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _batch_delete(s3, bucket: str, keys: list) -> tuple[int, int]:
    """Delete up to 1000 objects in a single S3/R2 API request.

    The delete_objects API accepts up to 1000 keys per call, making it
    ~1000x faster than calling delete_object individually.

    Returns:
        Tuple of (deleted_count, error_count).
    """
    response = s3.delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": k} for k in keys], "Quiet": True},
    )
    errors = response.get("Errors", [])
    for err in errors:
        print(f"  ERROR deleting {err['Key']}: {err['Message']}", flush=True)
    return len(keys) - len(errors), len(errors)


def move_objects(
    bucket: str,
    old_prefix: str,
    new_prefix: str,
    endpoint_url: str = None,
    copy: bool = False,
    delete: bool = False,
    delete_only: bool = False,
) -> dict:
    """Move or delete objects in an S3/R2 bucket.

    Supports three modes:
      - Dry run (default): list what would be moved
      - Copy: copy objects to new prefix, optionally delete originals
      - Delete only: remove objects under a prefix without copying

    Args:
        bucket: Bucket name.
        old_prefix: Source prefix to operate on. Use "" for all objects.
        new_prefix: Destination prefix (ignored when delete_only=True).
        endpoint_url: S3-compatible endpoint URL (e.g. Cloudflare R2).
        copy: If True, copy objects to new_prefix using a thread pool.
        delete: If True, delete originals after all copies succeed.
        delete_only: If True, delete objects under old_prefix without copying.

    Returns:
        dict with counts: total, copied, deleted, errors.
    """
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    paginator = s3.get_paginator("list_objects_v2")

    stats = {"total": 0, "copied": 0, "deleted": 0, "errors": 0}

    # --- DELETE ONLY MODE ---
    # Uses batch delete API: 1000 keys per request
    if delete_only:
        batch = []
        for page in paginator.paginate(Bucket=bucket, Prefix=old_prefix):
            for obj in page.get("Contents", []):
                batch.append(obj["Key"])
                stats["total"] += 1

                # Flush batch when it hits the 1000-key API limit
                if len(batch) == 1000:
                    deleted, errors = _batch_delete(s3, bucket, batch)
                    stats["deleted"] += deleted
                    stats["errors"] += errors
                    print(f"  deleted {stats['deleted']} objects...", flush=True)
                    batch = []

        # Delete any remaining objects in the last partial batch
        if batch:
            deleted, errors = _batch_delete(s3, bucket, batch)
            stats["deleted"] += deleted
            stats["errors"] += errors

        return stats

    # --- DRY RUN MODE ---
    # Print source -> destination mapping without making changes
    if not copy:
        for page in paginator.paginate(Bucket=bucket, Prefix=old_prefix):
            for obj in page.get("Contents", []):
                old_key = obj["Key"]
                new_key = new_prefix + old_key[len(old_prefix):]
                stats["total"] += 1
                print(f"  {old_key} -> {new_key}")
        return stats

    # --- COPY MODE ---
    # Uses a thread pool to copy objects concurrently (no bulk copy API exists)
    def _copy_one(old_key, new_key):
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": old_key},
            Key=new_key,
        )
        return old_key

    # First pass: collect all source/destination key pairs
    keys_to_copy = []
    for page in paginator.paginate(Bucket=bucket, Prefix=old_prefix):
        for obj in page.get("Contents", []):
            old_key = obj["Key"]
            new_key = new_prefix + old_key[len(old_prefix):]
            keys_to_copy.append((old_key, new_key))

    stats["total"] = len(keys_to_copy)
    print(f"  {stats['total']} objects to copy ({MAX_WORKERS} threads)...", flush=True)

    # Second pass: copy all objects using thread pool for concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_copy_one, old_key, new_key): old_key
            for old_key, new_key in keys_to_copy
        }
        for future in as_completed(futures):
            old_key = futures[future]
            try:
                future.result()
                stats["copied"] += 1
                if stats["copied"] % 1000 == 0:
                    print(f"  copied {stats['copied']} / {stats['total']}...", flush=True)
            except Exception as e:
                stats["errors"] += 1
                print(f"  ERROR {old_key}: {e}", flush=True)

    # Third pass (optional): batch delete originals after all copies complete
    if delete:
        print(f"  deleting {stats['copied']} originals...", flush=True)
        batch = []
        for old_key, _ in keys_to_copy:
            batch.append(old_key)
            if len(batch) == 1000:
                deleted, errors = _batch_delete(s3, bucket, batch)
                stats["deleted"] += deleted
                stats["errors"] += errors
                batch = []
        if batch:
            deleted, errors = _batch_delete(s3, bucket, batch)
            stats["deleted"] += deleted
            stats["errors"] += errors

    return stats


def main():
    parser = argparse.ArgumentParser(description="Move or delete objects within an R2/S3 bucket.")
    parser.add_argument("old_prefix", help="Source prefix (use '' for all objects)")
    parser.add_argument("new_prefix", nargs="?", default="", help="Destination prefix (not needed with --delete-only)")
    parser.add_argument("--copy", action="store_true", help="Perform the copy (default is dry run)")
    parser.add_argument("--delete", action="store_true", help="Delete originals after copy")
    parser.add_argument("--delete-only", action="store_true", help="Delete objects under old_prefix without copying")
    parser.add_argument("--bucket", help="Override bucket (default: AWS_S3_BUCKET env var)")
    args = parser.parse_args()

    # Validate flag combinations
    if args.delete and not args.copy:
        print("ERROR: --delete requires --copy")
        sys.exit(1)

    if args.delete_only and (args.copy or args.delete):
        print("ERROR: --delete-only cannot be combined with --copy or --delete")
        sys.exit(1)

    # Use --bucket flag if provided, otherwise fall back to env var
    bucket = args.bucket or os.getenv("AWS_S3_BUCKET")
    endpoint_url = os.getenv("S3_ENDPOINT_URL")

    if not bucket:
        print("ERROR: AWS_S3_BUCKET not set")
        sys.exit(1)

    # Determine and display operation mode
    if args.delete_only:
        mode = "DELETE ONLY"
    elif not args.copy:
        mode = "DRY RUN"
    elif args.delete:
        mode = "COPY + DELETE"
    else:
        mode = "COPY ONLY"

    print(f"Bucket: {bucket}")
    print(f"Mode: {mode}")
    print(f"Prefix: {args.old_prefix}")
    if not args.delete_only:
        print(f"To:     {args.new_prefix}")
    print()

    stats = move_objects(
        bucket=bucket,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
        endpoint_url=endpoint_url,
        copy=args.copy,
        delete=args.delete,
        delete_only=args.delete_only,
    )

    print()
    print(f"Total: {stats['total']}, Copied: {stats['copied']}, "
          f"Deleted: {stats['deleted']}, Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
