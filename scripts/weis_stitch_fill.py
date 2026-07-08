"""One-time WEIS -> data_im West stitch-fill (Phase 2 of the RTO West migration).

Materializes the pre-launch West history in storage: copies the WEIS
(``data/``) consolidated lmp/mtlf/mtrf rows into the ``im/`` consolidated
tables with ``BAA='SWPW'`` and ``source='weis'``, so the West BAA has a
continuous training series across the 2026-04-01 seam.

Two LMP cases:
  * Exact-name West nodes (``node_list.WEST_HUB_BA_NODES`` present in WEIS,
    ~40 of them, incl. all seam nodes) are copied straight through.
  * The RTO West aggregated hubs have no WEIS equivalent. The flagship
    forecast target ``SWPW_HUB`` is proxied by the per-interval mean LMP of
    every WEIS ``WACM*`` settlement location — a documented cross-seam
    approximation, tagged ``source='weis'`` like the rest of the stitch.

WEIS had no Day-Ahead market and no per-reserve-zone renewable forecast, so
da_lmp and rf_reserve_zone get no stitch (West values start at RTO launch).

Idempotent: re-running drops the prior ``source='weis'`` rows first, so it can
be re-run from raw ``data/`` if the seam treatment ever changes.

Run only when nothing else is writing the ``im/`` tables — this and
``upsert_im`` both do whole-object read-modify-write on the same parquets, so a
concurrent collection/backfill job would clobber one side. Stop the IM Modal
app (or wait for the backfill) first.

Usage:  python scripts/weis_stitch_fill.py [--dry-run]
"""

import argparse
import logging
import os
import sys

import pandas as pd
import polars as pl
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("weis_stitch_fill")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_collection_utils import IM_PREFIX, WEIS_PREFIX, _s3_storage_options  # noqa: E402
from data_collection_im import RTO_WEST_LAUNCH, UPSERT_KEYS  # noqa: E402
from node_list import WEST_HUB_BA_NODES  # noqa: E402

# WEIS was West-only real-time, so its system-wide load/resource forecasts
# and matched nodal prices are the West BAA series.
STITCH_BAA = "SWPW"
STITCH_SOURCE = "weis"
# Cap defensively at the seam so no stitched row overlaps the IM era.
SEAM = RTO_WEST_LAUNCH

# RTO West aggregated hubs have no exact WEIS match, so proxy each from the
# per-interval mean of its WEIS constituent nodes (name-prefix match). The
# proxy tracks the real post-launch node's price level; averaging smooths the
# congestion spikes, so it understates variance. The prefixes are domain
# mappings (RTO West node -> WEIS sub-entity), not derivable from the feeds.
PROXY_MAP = {
    "SWPW_HUB": "WACM",             # SPP West hub ~ WAPA Colorado-Missouri nodes
    "PSCO": "PSCO.PSCM.",          # Public Service Co of Colorado (PSCo market)
    "BHBA": "PSCO.BHCE.",          # Black Hills Colorado Electric
    "WACM_CRSP_WILW": "WACM.CRSP.",  # Colorado River Storage Project (WACM)
}


def _base_paths() -> tuple[str, str]:
    """Return (weis_base, im_base) S3 prefixes from the AWS env vars."""
    bucket = os.environ["AWS_S3_BUCKET"]
    folder = os.environ.get("AWS_S3_FOLDER", "")
    return f"s3://{bucket}/{folder}{WEIS_PREFIX}", f"s3://{bucket}/{folder}{IM_PREFIX}"


def _build_proxy(lmp: pl.LazyFrame, node: str, prefix: str) -> pl.DataFrame:
    """Proxy one hub from the per-interval mean of its WEIS constituents.

    Args:
        lmp: Pre-seam WEIS LMP LazyFrame.
        node: The RTO West node name to synthesize (e.g. 'PSCO').
        prefix: Settlement-location name prefix of its WEIS constituents
            (e.g. 'PSCO.PSCM.'); all matching nodes are averaged.

    Returns:
        pl.DataFrame of proxy rows tagged BAA='SWPW', source='weis'.
    """
    return (
        lmp.filter(pl.col("Settlement_Location_Name").str.starts_with(prefix))
        .group_by(["Interval_HE", "GMTIntervalEnd_HE", "timestamp_mst_HE"])
        .agg(
            # mean promotes Float32 -> Float64; cast back to match the table
            pl.col("LMP").mean().cast(pl.Float32),
            pl.col("MLC").mean().cast(pl.Float32),
            pl.col("MCC").mean().cast(pl.Float32),
            pl.col("MEC").mean().cast(pl.Float32),
        )
        .with_columns(
            pl.lit(node).alias("Settlement_Location_Name"),
            pl.lit(node).alias("PNODE_Name"),
            pl.lit(STITCH_BAA).alias("BAA"),
            pl.lit(pd.Timestamp.now("UTC").tz_localize(None)).alias("file_create_time_utc"),
            pl.lit(f"weis-stitch:mean({prefix}*)").alias("url"),
            pl.lit(STITCH_SOURCE).alias("source"),
        )
        .collect()
    )


def build_lmp_stitch(weis_base: str, so: dict) -> pl.DataFrame:
    """Build the West LMP stitch: exact-name matches + PROXY_MAP proxies.

    Reads the WEIS consolidated ``lmp.parquet``, keeps only intervals before
    the seam, and produces the West BAA rows two ways: exact-name West nodes
    copied straight through, and each aggregated hub in PROXY_MAP synthesized
    from the per-interval mean of its WEIS constituent nodes.

    Args:
        weis_base: S3 prefix of the WEIS ``data/`` tables.
        so: boto3/polars storage options for the R2 bucket.

    Returns:
        pl.DataFrame of stitch rows (matched nodes + proxies) with the same
        columns/dtypes as the data_im ``lmp`` table.
    """
    lmp = pl.scan_parquet(f"{weis_base}lmp.parquet", storage_options=so).filter(
        pl.col("GMTIntervalEnd_HE") < SEAM
    )

    matched = (
        lmp.filter(pl.col("Settlement_Location_Name").is_in(WEST_HUB_BA_NODES))
        .with_columns(
            pl.lit(STITCH_BAA).alias("BAA"),
            pl.lit(STITCH_SOURCE).alias("source"),
        )
        .collect()
    )
    n_matched = matched["Settlement_Location_Name"].n_unique()
    log.info(
        f"LMP stitch: {n_matched} exact-name West nodes, {matched.shape[0]:,} rows"
    )

    proxies = []
    for node, prefix in PROXY_MAP.items():
        px = _build_proxy(lmp, node, prefix).select(matched.columns)
        log.info(f"LMP stitch: {node} proxy over {prefix}* -> {px.shape[0]:,} rows")
        proxies.append(px)

    return pl.concat([matched, *proxies], how="vertical")


def build_forecast_stitch(weis_base: str, target: str, so: dict) -> pl.DataFrame:
    """Build the MTLF/MTRF stitch (WEIS system-wide == West BAA).

    Args:
        weis_base: S3 prefix of the WEIS ``data/`` tables.
        target: 'mtlf' or 'mtrf' — the consolidated table to read/stitch.
        so: boto3/polars storage options for the R2 bucket.

    Returns:
        pl.DataFrame of pre-seam WEIS rows tagged BAA='SWPW',
        source='weis'.
    """
    return (
        pl.scan_parquet(f"{weis_base}{target}.parquet", storage_options=so)
        .filter(pl.col("GMTIntervalEnd") < SEAM)
        .with_columns(
            pl.lit(STITCH_BAA).alias("BAA"),
            pl.lit(STITCH_SOURCE).alias("source"),
        )
        .collect()
    )


def merge_into_target(
    im_base: str, target: str, stitch: pl.DataFrame, so: dict, dry_run: bool
) -> None:
    """Replace the source='weis' rows of a data_im table with `stitch`.

    Reads the consolidated data_im table, drops any prior stitch rows
    (so re-runs are idempotent), appends the new stitch, and fails loud
    if the merge introduces duplicate upsert keys.

    Args:
        im_base: S3 prefix of the ``im/`` tables.
        target: Consolidated table name ('lmp', 'mtlf', 'mtrf').
        stitch: Stitch rows to merge, from build_lmp_stitch /
            build_forecast_stitch.
        so: boto3/polars storage options for the R2 bucket.
        dry_run: If True, log the row counts but do not write.

    Returns:
        None — writes the merged parquet unless dry_run is set.

    Raises:
        RuntimeError: If the merged table has duplicate upsert keys.
    """
    path = f"{im_base}{target}.parquet"
    existing = pl.read_parquet(path, storage_options=so)
    # Tolerate a pre-source-column table (validation-era rows are all IM).
    if "source" not in existing.columns:
        existing = existing.with_columns(pl.lit("im").alias("source"))

    kept = existing.filter(pl.col("source") != STITCH_SOURCE)
    merged = pl.concat([kept, stitch.select(existing.columns)], how="vertical")
    dups = merged.select(UPSERT_KEYS[target]).is_duplicated().sum()
    if dups:
        raise RuntimeError(f"{target}: {dups} duplicate upsert keys after stitch merge")

    log.info(
        f"{target}: existing={existing.shape[0]:,} (im={kept.shape[0]:,}) "
        f"+ stitch={stitch.shape[0]:,} -> merged={merged.shape[0]:,}"
    )
    if dry_run:
        log.info(f"{target}: --dry-run, not writing")
        return
    merged.write_parquet(path, storage_options=so)
    log.info(f"{target}: wrote {path}")


def main() -> None:
    """Run the one-time WEIS -> data_im West stitch-fill.

    Stitches the LMP table (matched nodes + SWPW_HUB proxy) and the
    MTLF/MTRF forecast tables. Pass ``--dry-run`` to report row counts
    without writing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="compute and report without writing"
    )
    args = parser.parse_args()

    so = _s3_storage_options()
    weis_base, im_base = _base_paths()

    merge_into_target(im_base, "lmp", build_lmp_stitch(weis_base, so), so, args.dry_run)
    for target in ("mtlf", "mtrf"):
        merge_into_target(
            im_base,
            target,
            build_forecast_stitch(weis_base, target, so),
            so,
            args.dry_run,
        )

    log.info("WEIS stitch-fill complete")


if __name__ == "__main__":
    main()
