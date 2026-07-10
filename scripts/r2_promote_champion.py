"""Promote (or revert to) a retrained model by repointing champion.json.

The Shiny app decides which model to serve by reading
``models/champion.json`` from R2 and loading the checkpoints in that
JSON's ``champion_artifact_folder``. Retrains land in timestamped
``models/retrains/<timestamp>/`` folders; ``model_retrain.py`` only writes
champion.json when ``PROMOTE_CHAMPION`` is true. This script is the manual
path: point the live champion at any existing ``models/retrains/`` folder —
to promote a model staged with ``PROMOTE_CHAMPION=false``, or to revert to a
prior model by naming its older folder.

It never moves or copies checkpoints (they already live in their timestamped
folder); it only rewrites the small champion.json pointer, exactly matching
the three-key schema ``model_retrain.py`` writes. The app picks up the change
on its next data/model reload.

Usage:
    # List the models/retrains/ folders available to promote (newest last)
    python scripts/r2_promote_champion.py --list

    # Show the model champion.json currently points at
    python scripts/r2_promote_champion.py --show

    # Dry run (default): show current -> target without writing
    python scripts/r2_promote_champion.py 2026-07-06_12-41-45

    # Actually repoint champion.json at that folder
    python scripts/r2_promote_champion.py 2026-07-06_12-41-45 --promote
"""

import argparse
import json
import os
import sys

import boto3
from dotenv import load_dotenv

# Load .env with override so it takes precedence over any existing env vars
# (matches scripts/r2_move_objects.py).
load_dotenv(override=True)

# The R2 storage-layout constants live in src/utils. scripts/ run as
# standalone files, so put src/ on the path like scripts/weis_stitch_fill.py
# does (utils is light — boto3/json/os only, no darts).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import (  # noqa: E402
    build_champion_config,
    champion_key_suffix,
    retrains_prefix,
)

# Default forecast target (parameters.TARGETS canonical set; 'da' is the primary
# model). Kept as a literal so this CLI stays darts-free like the rest of the
# script — pass --target rt to operate on the real-time model.
DEFAULT_TARGET = "da"


def make_s3_client():
    """Build an R2/S3 client from the environment.

    Returns:
        A boto3 S3 client pointed at S3_ENDPOINT_URL (Cloudflare R2), matching
        the client construction used across src/utils.py.
    """
    return boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))


def list_retrain_folders(s3, bucket: str, folder: str, retr_prefix: str) -> list[str]:
    """List the retrain timestamp folders under the target's retrains prefix.

    Args:
        s3: Boto3 S3 client.
        bucket: Bucket name.
        folder: AWS_S3_FOLDER prefix ("" or trailing-slash prefix).
        retr_prefix: The target's retrains prefix (e.g. "models/da/retrains/").

    Returns:
        Timestamp folder names (e.g. "2026-07-06_12-41-45"), sorted ascending
        so the newest retrain is last. The lexical sort works because the
        folders are "%Y-%m-%d_%H-%M-%S" stamps.
    """
    prefix = folder + retr_prefix
    paginator = s3.get_paginator("list_objects_v2")
    names = set()
    # Delimiter="/" makes S3 return the immediate subfolders as CommonPrefixes
    # instead of every checkpoint object under them.
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            names.add(cp["Prefix"][len(prefix) :].rstrip("/"))
    return sorted(names)


def folder_has_objects(s3, bucket: str, key_prefix: str) -> bool:
    """Check that an R2 prefix contains at least one object.

    Guards against pointing the live champion at an empty or misspelled
    folder, which would break model loading in the app.

    Args:
        s3: Boto3 S3 client.
        bucket: Bucket name.
        key_prefix: Full object-key prefix to test (AWS_S3_FOLDER included).

    Returns:
        True if one or more objects exist under the prefix.
    """
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def get_current_champion(s3, bucket: str, champion_key: str) -> dict | None:
    """Read the current champion.json, or None if it does not exist yet.

    Args:
        s3: Boto3 S3 client.
        bucket: Bucket name.
        champion_key: Full key of champion.json.

    Returns:
        The parsed champion config dict, or None if champion.json is absent.
    """
    try:
        resp = s3.get_object(Bucket=bucket, Key=champion_key)
    except s3.exceptions.NoSuchKey:
        return None
    return json.loads(resp["Body"].read().decode("utf-8"))


def build_champion_json(folder: str, timestamp: str, retr_prefix: str) -> dict:
    """Build the champion.json payload for a target retrain folder.

    Derives the path pieces from the CLI timestamp and delegates the schema
    to utils.build_champion_config, so a promoted-by-script model is
    identical to a promoted-by-retrain one (same single source of the schema).

    Args:
        folder: AWS_S3_FOLDER prefix ("" or trailing-slash prefix).
        timestamp: The retrain folder name (e.g. "2026-07-06_12-41-45").
        retr_prefix: The target's retrains prefix (e.g. "models/da/retrains/").

    Returns:
        The champion config dict with champion / champion_artifact_folder /
        champion_artifact_path keys.
    """
    folder_time = timestamp + "/"
    artifact_folder = retr_prefix + folder_time
    return build_champion_config(folder_time, artifact_folder, folder + artifact_folder)


def main() -> int:
    """Parse arguments and promote, list, or show the champion pointer.

    Returns:
        Process exit code (0 on success, non-zero on a usage or validation
        error).
    """
    parser = argparse.ArgumentParser(
        description="Promote or revert the served model by repointing champion.json.",
    )
    parser.add_argument(
        "timestamp",
        nargs="?",
        help="Retrain folder to promote, e.g. 2026-07-06_12-41-45 "
        "(the folder name under models/retrains/). Omit with --list/--show.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Actually write champion.json. Without this flag the script is a "
        "dry run that only prints the current -> target change.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the models/retrains/ folders available to promote and exit.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the current champion.json and exit.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Forecast target to operate on (models/<target>/); default {DEFAULT_TARGET!r}.",
    )
    args = parser.parse_args()

    bucket = os.getenv("AWS_S3_BUCKET")
    folder = os.getenv("AWS_S3_FOLDER", "")
    if not bucket:
        print("ERROR: AWS_S3_BUCKET is not set (check your .env).", flush=True)
        return 1

    retr_prefix = retrains_prefix(args.target)
    s3 = make_s3_client()
    champion_key = folder + champion_key_suffix(args.target)

    if args.list:
        folders = list_retrain_folders(s3, bucket, folder, retr_prefix)
        if not folders:
            print(f"No retrain folders under {folder + retr_prefix}", flush=True)
            return 0
        print(f"Available retrains under {folder + retr_prefix} (newest last):")
        for name in folders:
            print(f"  {name}")
        return 0

    current = get_current_champion(s3, bucket, champion_key)
    if args.show:
        if current is None:
            print(f"No champion.json at {champion_key}", flush=True)
        else:
            print(json.dumps(current, indent=2))
        return 0

    if not args.timestamp:
        parser.error("a timestamp is required unless --list or --show is given")

    # Normalize: accept a bare timestamp or a "models/<target>/retrains/<ts>/" path.
    timestamp = args.timestamp.strip("/")
    if timestamp.startswith(retr_prefix):
        timestamp = timestamp[len(retr_prefix) :].strip("/")

    target = build_champion_json(folder, timestamp, retr_prefix)
    target_prefix = target["champion_artifact_path"]

    # Refuse to point the live app at a folder that has no checkpoints.
    if not folder_has_objects(s3, bucket, target_prefix):
        print(
            f"ERROR: no objects under {target_prefix} — "
            f"'{timestamp}' is not a valid retrain folder. "
            "Run with --list to see available folders.",
            flush=True,
        )
        return 1

    current_folder = (current or {}).get("champion_artifact_folder", "(none)")
    print(f"champion.json: {champion_key}")
    print(f"  current champion_artifact_folder: {current_folder}")
    print(f"  target  champion_artifact_folder: {target['champion_artifact_folder']}")

    if not args.promote:
        print("\nDry run — pass --promote to write this change.", flush=True)
        return 0

    body = json.dumps(target).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=champion_key, Body=body)
    print(f"\nPromoted. Wrote champion.json -> {target['champion_artifact_folder']}")
    print("The app serves the new model on its next data/model reload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
