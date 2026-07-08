# Migration Plan: SPP WEIS → SPP RTO West (Integrated Marketplace)

> **✅ Migration complete and live (status as of 2026-07-08).** Phases 0–4
> shipped: the app forecasts RTO West / Integrated Marketplace prices on the
> curated 10-node West set, with the IM-tuned `spp_west` champion promoted and
> live. All design decisions are resolved. The Optuna re-tune once listed as a
> "next action" is **done** — the single-objective CRPS study replaced the
> WEIS-tuned params (harness CRPS 61.4→17.2). The only remaining work is the
> **optional** polish in the "Next actions" list below (app-map node geometry,
> Phase 5 code cleanup, an R2 bucket rename, one reserve-zone decision) — none
> are migration blockers.

## Current state (updated 2026-07-05)

> **Update (2026-07-07) — supersedes the stitch / break-indicator design below.**
> The **STITCH** strategy (glue WEIS history onto IM with a break-indicator
> covariate + 365-day window) was **not adopted**. Training is **IM-only**,
> clamped to the 2026-04-01 RTO West launch: `de._default_start_time()` =
> `max(now − TRAIN_START, RTO_WEST_LAUNCH)`. The window grows from the launch
> until ~2027-04, then becomes a rolling 365-day window. Rationale: WEIS-era
> prices are a much calmer regime, so mixing them biased the model toward flat
> forecasts.
>
> Consequences:
> - The `break_indicator` future covariate (added in Phase 4) was **removed
>   2026-07-07** — IM-only training makes it a degenerate constant 1. `FUTR_COLS`
>   is now 11 covariates.
> - A model-improvements effort shipped and is live: **Darts 0.41→0.45**, a
>   **single-objective CRPS re-tune** (harness CRPS 61.4→17.2), **wider tail
>   quantiles** (`parameters.QUANTILES`, coverage 0.85→0.88), and a
>   **`training_config.json`** saved with every model (covariates / nodes /
>   quantiles / versions / train window) that the app **validates at load** —
>   a covariate mismatch now fails loud with a clear message.
> - **Modal retrain job:** `spp-weis-model-retrain` (`modal_jobs/model_retrain.py`,
>   weekly Sun 20:00 UTC, auto-promotes). It **bakes code in at deploy time and
>   does not auto-pull** — redeploy with `modal deploy modal_jobs/model_retrain.py`
>   after code changes. (Not currently a live Modal app.)
>
> The stitch / break-indicator sections below are kept for history but are
> superseded by the above.

Where things stand on `feature/rto-west-migration`, for picking up in a fresh session:

**Done**
- **Phase 0** feed/schema verification for all five core feeds (the table below was
  re-verified with live pulls on 2026-07-05; the daily LMP rollup joined it last).
- Plan reviewed and corrected: West filtering moved downstream (store both BAAs at
  collection), daily-LMP rollup flagged unverified, gen-capacity dropped as dead code,
  PCM coverage numbers derived from live data.
- **Plan re-review + interview (2026-07-05):** code touch points verified against source;
  decisions amended — **LMP storage narrowed to hub/BA node rows only (both BAAs)**,
  **`BAA` added to every upsert dedup key** (clobber bug otherwise), break indicator
  pinned as a *future* covariate with a drop date, and the daily-LMP question given a
  resolution path (listing-API search, else widen the 5-min re-pull). Details inline
  below and in "Decisions locked".
- **Daily-LMP question RESOLVED (2026-07-05):** the IM daily rollup exists at the
  WEIS-analogous `By_Day` path — the earlier 404s were its **5-day publication lag**.
  Collector kept (lag-aware window); the Phase 2 East LMP backfill uses the pre-launch
  daily files (~365 pulls, not ~105k). See Phase 0.
- **App copy updated** (`app.py`, `src/plotting.py`): WEIS labels/links → IM West
  equivalents. This is the app half of Phase 3; the settlement-location universe still
  pends the Phase 3 data-engineering work.
- **Node-geometry prototypes** copied into `scripts/node_geometry_prototype/` (seeds for
  Phase 3b `src/geometry.py`).
- Dev tooling landed: `.claude/` skills (`code-quality`, `comment-docstring`,
  `security-scan`, `simplify-audit`) + `code-reviewer`/`simplify-auditor` agents, repo
  `CLAUDE.md`, ruff per-file ignores, detect-secrets baseline, and `.env.example`
  documenting the R2 env keys. The R2 bucket is **`spp-weis-forecast`**.

**Done (continued)**
- **Phase 1 collector code built** (`src/data_collection_im.py` + `src/node_list.py`): all
  six collectors (RTBM 5-min LMP, daily LMP with the 5-day-lag-aware window, MTLF, MTRF,
  `RF_RESERVE_ZONE`, DA LMP — slug verified as `da-lmp-by-settlement-location`), DST
  `…d.csv` handling, pre-launch missing-`BAA` tolerance, LMP filtered to the hub/BA node
  list at storage, and `BAA` in every upsert dedup key. Unit-tested against real trimmed
  portal CSVs (`tests/unit/test_data_collection_im.py`, `tests/unit/fixtures/`), passed
  the `code-reviewer` gate, and **validated end-to-end live**: one small collection ran
  through all six feeds into `data_im/` on R2 — all five consolidated tables written with
  both BAAs, all stored nodes present, zero dedup-key duplicates.

- **Phase 1 wrap-up wired** (`notebooks/data_collection/data_collection_im_hourly.py`,
  `data_collection_im_daily.py` + `modal_jobs/data_collection_im.py`, Modal app
  `spp-im-data-collection`): thin marimo notebooks calling the IM collectors, wrapped by
  two Modal jobs — `collect_im_hourly` (MTLF/MTRF/RF/5-min LMP, every 4h) and
  `collect_im_daily` (daily-LMP repair sweep + DA LMP, every 3 days). Both **run live end-to-end**
  as scripts (2026-07-05): all five `data_im/` tables written, both BAAs, all 10 East hubs
  present, zero dedup-key duplicates. **Deployed 2026-07-05** (`modal deploy`); the WEIS
  `spp-weis-data-collection` jobs were **stopped** (feeds dead since 2026-04-01). A `source`
  column (`'im'`/`'weis'`) was added to the IM writes for stitch provenance.

- **Phase 2 backfill + stitch done** (`notebooks/data_collection/data_collection_im_backfill.py`
  + `scripts/weis_stitch_fill.py`, committed): backfilled MTLF/MTRF/RF + LMP + DA 2025-04-01 →
  present and materialized the WEIS West stitch. Validated live: `lmp` 1,021,935 rows, both
  BAAs, `im`/`weis` source split, 0 dup keys; `SWPW_HUB` continuous 886 d across the seam.
  Backfill surfaced+fixed three robustness bugs (whole-file schema inference, one flexible
  datetime parser for all feeds, skip-and-log malformed files with a batch-failure warning).

- **Phase 3 data engineering + app + tests done** (`src/data_engineering.py`, committed):
  `create_database` reads `data_im/`; `prep_lmp`/`prep_mtlf`/`prep_mtrf` filter `BAA=='SWPW'`
  and `prep_lmp` swaps `loc_filter='PSCO_'` for `node_list.WEST_HUB_BA_NODES` (dropping the
  WEIS `_ARPA` exclusion). App labels/links already read "SPP IM West"; the location universe
  flows from `prep_lmp` so it auto-updates. Unit + e2e fixtures repaired (BAA column, West
  node names, IM West title). Validated live: `prep_lmp` → 64 West nodes over a 365-day window
  across the seam; West-only `MTLF` (not whole-RTO). `ReserveZone==21` is deferred — RF is not
  a covariate yet (see open question below).

- **Phase 4 done + LIVE (2026-07-06):** added the 2026-04-01 **break-indicator future
  covariate** (`RTO_WEST_LAUNCH` single-homed in `node_list`) — *removed 2026-07-07, see
  update above_ — set `MODEL_NAME='spp_west'`,
  scoped modeling/app to a curated **10-node `MODEL_APP_NODES`** (8 internal West + `BPA`/`CISO`
  — the 25 seam interfaces are one near-identical CAISO/WECC signal, so only two are kept),
  retrained a 5-member TiDE ensemble locally on GPU (~10 s/epoch, val RMSE ≈ $9.9), and
  **promoted** the `spp_west` champion. **Merged to `main`** (`0745d79`) → Posit Connect
  auto-deploys the new West serving code; e2e validated the promoted champion + new code are
  compatible (all 6 pass). Revert = point `champion.json` back to `2026-07-05_20-08-55/`.
  Optuna re-tune was deferred at Phase 4 (quick-retrain-first) but has **since
  been done** — the single-objective CRPS study replaced the WEIS-tuned
  `TIDE_PARAMS` with IM-tuned ones (harness CRPS 61.4→17.2).

**Data-quality finding (2026-07-05) — stitch continuity by exact node name:**
Checking the 64 West nodes against the WEIS history: **seam 25/25 present, internal only
15/39**. The 24 missing internal nodes include the flagship target **`SWPW_HUB`**, `PSCO`,
`CRSP_HUB`, `LAP_HUB`, and the aggregate `.FSE` constructs — because WEIS priced *granular
pnodes* (`PSCO.*`, `WACM.*`) while RTO West introduced *aggregated* hubs with no exact WEIS
name. **Resolution:** `scripts/weis_stitch_fill.py` carries a `PROXY_MAP` that proxies each
aggregated hub from the per-interval mean of its WEIS constituents (domain-mapped prefix),
tagged `source='weis'`: `SWPW_HUB←mean(WACM*)`, `PSCO←mean(PSCO.PSCM.*)`,
`BHBA←mean(PSCO.BHCE.*)`, `WACM_CRSP_WILW←mean(WACM.CRSP.*)`. Proxy levels match the real
post-launch nodes within ~$1–4; averaging smooths the congestion spikes (understates
variance). The 6 nodes originally suspected of missing data (`DEAA/DOPD/EPE/GCPD/GRID/GWA`)
are in fact fully covered on both sides of the seam.

**Next actions (optional; the migration is live)**
1. ~~**Optuna re-tune**~~ — ✅ **DONE.** The single-objective CRPS study on the IM-only West
   data replaced the WEIS-tuned `TIDE_PARAMS` with IM-tuned ones and re-promoted (harness
   CRPS 61.4→17.2, coverage ~0.88). The stitch was not used (see the 2026-07-07 update above).
2. **Phase 3b — node geometry**: build `src/geometry.py::fetch_pcm_geometries()` +
   `src/reference/node_geometry.csv` + a refresh notebook (map coordinates for the app).
3. ~~**Phase 5 cleanup** — extract shared collection helpers~~ — ✅ **DONE (2026-07-08).**
   The eight feed-agnostic helpers (`N_JOBS`, `ProgressParallel`, `_s3_storage_options`,
   `add_timestamp_mst`, `check_file_exists_client`, `format_df_colnames`, `get_csv_from_url`,
   `set_he`) now have a single home in `src/data_collection_utils.py`. The live IM collector
   (`data_collection_im.py`) and the stitch script import from it; `data_collection.py` re-imports
   them for its WEIS-specific feed logic. Tests split into `tests/unit/test_data_collection_utils.py`.
   `data_collection.py` stays in `src/` for now; fully retiring it to `deprecated/` is a separate
   cluster move (the module + the unwired weather notebook + `test_data_collection.py`).
   Still optional: R2 bucket rename `spp-weis-forecast`→`spp-im-bucket`.
4. Open decision: whether West `RF_RESERVE_ZONE` (zone 21, post-launch only) is worth adding
   as a covariate (its `ReserveZone==21` filter is ready to wire in if so).

## Background / why this is needed

SPP's **Western Energy Imbalance Service (WEIS)** — the real-time-only transitional
market this project forecasts — was **permanently terminated on April 1, 2026**. On the
same date SPP launched full **RTO operations in the Western Interconnection** ("RTO West"),
absorbing all former WEIS participants into the SPP **Integrated Marketplace (IM)** — the full
RTO market SPP now runs across both interconnections. Throughout this plan **IM** denotes the new
Integrated Marketplace feeds/artifacts (SPP's own term), in contrast to the legacy WEIS feeds. SPP is
the first US grid operator to run organized markets across both interconnections, now
operating two balancing authority areas (BAAs): **SPP East** and **SPP West** (the West BAA
is reported as `SWPW` / `SPPISO-West`).

**Consequences for this project:**

1. All WEIS data feeds (`portal.spp.org/file-browser-api/download/*-weis`, files prefixed
   `WEIS-`) stopped publishing new data after 2026-04-01. **Data collection is currently
   dead** — the hourly/daily Modal jobs are fetching URLs that no longer receive updates.
2. The replacement data lives in the **Integrated Marketplace** feeds (no `-weis` slug, no
   `WEIS-` filename prefix). These feeds now include **both** East and West nodes,
   distinguished by a new **BAA column**. We must filter to the West BAA.
3. The market design changed materially (a **Day-Ahead Market** now exists alongside RTBM;
   new resources, new settlement locations, ~300 new tradable West nodes). The forecast
   target's statistical behavior will shift, so **the model must be retrained on RTO West
   data** — and there is a hard data discontinuity at 2026-04-01.

Sources:
[SPP RTO West launch (Apr 1 2026)](https://kilowattlogic.com/news/spp-rto-west-launches-april-2026-dual-interconnection),
[SPP RTO Expansion](https://www.spp.org/western-services/rto-expansion/),
[Yes Energy – Preparing for SPP's RTO Expansion](https://www.yesenergy.com/blog/preparing-for-spps-rto-expansion),
[SPP Western Services](https://www.spp.org/western-services/),
[RTBM LMP by settlement location](https://portal.spp.org/pages/rtbm-lmp-by-location).

---

## What changes conceptually (WEIS → IM)

| Concept | WEIS (old) | RTO West / Integrated Marketplace (new) |
|---|---|---|
| Market scope | West-only, real-time only | Full RTO, Day-Ahead + Real-Time (RTBM), East **and** West |
| Endpoint slug | `…-weis` (e.g. `lmp-by-settlement-location-weis`) | no suffix (e.g. `rtbm-lmp-by-location`) |
| CSV filename prefix | `WEIS-RTBM-LMP-SL-…`, `WEIS-OP-MTLF-…`, `WEIS-OP-MTRF-…` | `RTBM-LMP-SL-…`, `OP-MTLF-…`, `OP-MTRF-…` |
| Geographic filter | implicit (all WEIS = West) | **explicit BAA filter required** (`SWPW`/`SPPISO-West`) |
| Load forecast (MTLF) | system-wide (== West) | published **per BAA**; take West BAA only |
| Resource forecast (MTRF) | West wind/solar | per BAA; take West |
| DST files | single file per interval | interval files + a `…d.csv` duplicate-hour variant on fall-back |

> ⚠️ **Semantic subtlety:** the old WEIS MTLF/MTRF were inherently West-only. The IM
> "system-wide" objects are being retired in favor of BAA-level objects (`SPPISO-East` /
> `SPPISO-West`). We must select the **West BAA** rows or the load/resource covariates will
> silently become whole-RTO aggregates and corrupt the model.

---

## Phase 0 — feeds & schema (VERIFIED 2026-07-05)

All IM feeds were confirmed by pulling live sample files from `portal.spp.org`. The IM CSVs
are **identical to the WEIS CSVs plus a trailing `BAA` column**; West rows are `BAA == 'SWPW'`
(East is `SPP`). Confirmed old→new mapping:

| Feed | IM slug | Path | Filename | Columns (new = `BAA`) |
|---|---|---|---|---|
| RTBM 5-min LMP | `rtbm-lmp-by-location` | `/{Y}/{M}/By_Interval/{D}/` | `RTBM-LMP-SL-{YYYYMMDDHHMM}.csv` | `Interval,GMTIntervalEnd,Settlement Location,Pnode,LMP,MLC,MCC,MEC,BAA` |
| Load (MTLF) | `mtlf-vs-actual` | `/{Y}/{M}/{D}/` | `OP-MTLF-{YYYYMMDDHH}00.csv` | `Interval,GMTIntervalEnd,MTLF,Averaged Actual,BAA` |
| Wind/Solar (MTRF) | `midterm-resource-forecast` | `/{Y}/{M}/{D}/` | `OP-MTRF-{YYYYMMDDHH}00.csv` | `Interval,GMTIntervalEnd,Wind Forecast MW,Solar Forecast MW,BAA` |
| Resource by Reserve Zone | `resource-forecast-by-reserve-zone` | `/{Y}/{M}/{D}/` | `RF_RESERVE_ZONE-{YYYYMMDDHH}00.csv` | `IntervalEnd,GMTIntervalEnd,BAA,ReserveZone,WindForecastMW,WindActualMW,SolarForecastMW,SolarActualMW` |
| RTBM daily LMP rollup | `rtbm-lmp-by-location` | `/{Y}/{M}/By_Day/` | `RTBM-LMP-DAILY-SL-{YYYYMMDD}.csv` (**publishes at D+5 ~18:00**) | `Interval,GMT Interval,Settlement Location Name,PNODE Name,LMP,MLC,MCC,MEC,BAA` |

- **Column parity:** identical to WEIS after `format_df_colnames`, so existing processors work
  almost verbatim — the real deltas are (1) URL slug, (2) `WEIS-` prefix gone → rework the
  `url.split('WEIS-')` filename parse, (3) keep the new `BAA` column at collection; **LMP
  rows are scoped to the hub/BA node list at storage** (amended 2026-07-05), the other
  feeds are stored whole with the `BAA == 'SWPW'` West filter applied **downstream** in
  data engineering (see Decisions).
- **Blank-`BAA` rows:** live files (esp. MTRF and `RF_RESERVE_ZONE`) carry leading rows with
  empty `BAA` and empty forecast values — future intervals not yet populated. The downstream
  `BAA == 'SWPW'` filter drops them naturally; include such rows in test fixtures.
  (`RF_RESERVE_ZONE` headers also have leading spaces — `format_df_colnames` already strips them.)
- **Pre-launch files have NO `BAA` column** (verified 2026-07-05 on 2025-07 samples): the same
  slugs/filenames publish East-only IM data going back years, with the `BAA` column added only
  at RTO West launch. Processors must tolerate the missing column and fill `BAA='SPP'` for
  pre-2026-04-01 files (matters for the Phase 2 East backfill).
- **Timestamps** remain Central-time-based (SPP operates on CPT); the `America/Chicago` ceil
  and `-7h` MST offset are still correct.
- **DST:** interval LMP files add a `…d.csv` duplicate-hour variant on fall-back — handle it.
- **West node universe:** ~302 distinct `SWPW` settlement locations (vs 348 WEIS). Only **42
  match WEIS by exact name** (~12%); the rest are renamed or new (see Stitching, below).
- **Resource by Reserve Zone** (`RF_RESERVE_ZONE`, hourly, +7 days) is **included** — it is
  the only feed carrying wind/solar **actuals** (MTRF has forecasts only). Store **all** zones
  (consistent with both-BAA storage); **`ReserveZone == 21`** (= the entire West BAA) is a
  downstream filter. Note the West is a *single* reserve zone, so this adds **no sub-BAA
  geographic detail** — its value is the actuals, not finer geography.
- **Daily LMP rollup — VERIFIED 2026-07-05 (via the listing API).** The feed exists at
  exactly the WEIS-analogous path: `/{Y}/{M}/By_Day/RTBM-LMP-DAILY-SL-{YYYYMMDD}.csv` under
  `rtbm-lmp-by-location`. The earlier 404s were a **~5-day publication lag**: the file for
  operating day D lands at ~18:00 on D+5 (e.g. 06-01 published 06-06; on 07-05 the newest
  file was 06-29), so probing recent dates always 404s. **Keep the daily collector** (it's
  the trailing repair sweep into the same consolidated `lmp` table — 7 requests/run vs
  ~2,000 to replay from 5-min files) but **offset its window by the lag** (pull days ending
  at `end_ts - 5d`, not `end_ts`). Schema matches the 5-min feed: post-launch files carry
  the trailing `BAA` column; **pre-launch files exist** (e.g. 31 files in `/2025/07/By_Day/`)
  without `BAA` — so the Phase 2 East-era LMP backfill can use **~365 daily pulls (~47 MB
  each) instead of ~105k 5-min pulls**. (The slug also has a `RePrice/` folder — corrected
  LMP republications; not collected today, noted for a future repair-sweep upgrade.)
- **File-browser listing API (used for the verification, handy for future ones):**
  `GET https://portal.spp.org/file-browser-api/?fsName={slug}&path={path}&type=folder`
  returns a JSON listing (name, path, size, modified) with no auth — e.g.
  `?fsName=rtbm-lmp-by-location&path=/2026/06/By_Day`.
- **Other granular forecasts not used:** STLF (5-min load, ±10 min) and STRF (5-min wind/solar,
  +4 h) — horizons far too short for the 120-hour price forecast.
- **Gen-capacity-by-fuel: DROPPED.** `get_gen_cap_url` is dead code — nothing calls it (the
  Modal jobs don't collect it, and `prep_gen_cap` is referenced only by tests). No IM slug
  needed; delete the collector + tests during the sweep.

---

## Code touch points (concrete)

`grep` finds ~70 `weis`/`WEIS` references across 20 files. The load-bearing ones:

**1. `src/data_collection.py` — URL builders (the core change).**
Five URL builders hardcode WEIS slugs + `WEIS-` prefixes, and four processors parse filenames
via `url.split('WEIS-')[-1]`:
- `get_hourly_mtlf_url`, `get_hourly_mtrf_url`, `get_5min_lmp_url`, `get_daily_lmp_url` —
  swap base URLs/paths to IM feeds (`get_daily_lmp_url`: verified 2026-07-05, same By_Day
  path; the daily *collector window* must account for the **5-day publication lag** —
  see Phase 0). `get_gen_cap_url` is **deleted**, not migrated (dead code — see Phase 0).
- `get_process_mtlf` / `get_process_mtrf` / `get_process_5min_lmp` / `get_process_daily_lmp`
  — the `url.split('WEIS-')[-1]` filename parsing breaks (no `WEIS-` prefix); rework to the
  new prefix. Keep the `BAA` column, both BAAs. **LMP processors filter rows to the hub/BA
  node list at storage** (amended 2026-07-05 — the node list lives in ONE home,
  `src/parameters.py` or a `src/reference/` file, shared with data engineering); MTLF/MTRF
  are stored whole, with West filtering downstream (see Decisions).
- **`upsert_mtlf_mtrf_lmp` dedup keys MUST gain `BAA`.** The mtlf/mtrf key is currently
  `GMTIntervalEnd` alone — with both BAAs stored, East and West rows for the same interval
  clobber each other (the upsert keeps whichever file wrote last, silently dropping one
  BAA). New keys: mtlf/mtrf → `(GMTIntervalEnd, BAA)`; lmp → add `BAA` alongside
  `(GMTIntervalEnd_HE, Settlement_Location_Name, PNODE_Name)`; new `rf_reserve_zone` →
  `(GMTIntervalEnd, BAA, ReserveZone)`.
- Handle the new DST `…d.csv` filename variant — the fix lives in the **range generation**
  (`get_range_data` must emit one extra fetch for the duplicated hour on fall-back days),
  not just the URL builders; `GMTIntervalEnd` disambiguates the rows downstream.
- Column mapping: confirm IM CSVs still expose `Settlement_Location`/`Pnode`/`LMP/MLC/MCC/MEC`
  and `MTLF`/`Averaged_Actual`/`Wind_Forecast_MW`/`Solar_Forecast_MW`, or update the renames
  and `.cast()`s accordingly.
- **New collector — Resource by Reserve Zone** (`RF_RESERVE_ZONE`): no WEIS equivalent. Add a
  URL builder + processor (store **all** zones; `ReserveZone == 21` = West is a downstream
  filter), a consolidated `rf_reserve_zone.parquet` target, and wire it into the hourly job —
  it supplies wind/solar **actuals**.
- **New collector — Day-Ahead LMP** (slug verified: `da-lmp-by-settlement-location`): collect
  to `data_im/da_lmp/` (both BAAs, hub/BA node rows only — same LMP storage rule) for history
  accrual. Files mix timestamp formats with/without seconds — handled by
  `convert_datetime_cols` (the single flexible parser now shared by every IM
  feed). **Not** consumed by the model yet
  (deferred); reserved for a future RT covariate or standalone DA forecasting model.

**2. `src/data_engineering.py` — location filtering & feature build.**
- **The West-BAA filter lands here** for the whole-stored feeds: `BAA == 'SWPW'` in the
  MTLF / MTRF prep and `ReserveZone == 21` for the reserve-zone feed. This also drops the
  blank-`BAA` rows present in live files. LMP arrives already scoped to the hub/BA node
  list (amended 2026-07-05); selecting the West model universe from it is still a
  downstream `BAA`/node-list step.
- `proc_lmp()` filters `Settlement_Location_Name` by `loc_filter` and drops `_ARPA`; the West
  node naming convention may differ — revisit `loc_filter` and the `_ARPA` exclusion. The
  hub/BA **node list gets one home** (`src/parameters.py` or `src/reference/`) shared by the
  collector-side filter and `proc_lmp` — don't declare it twice.
- `unique_id` universe (line ~461) is derived from whatever LMP data is present — will
  auto-populate from West nodes once collection is fixed, but the model's trained id set won't
  match until retrain.
- Timezone: `timestamp_mst` (GMT `-7h`) is fine for the West; keep, but re-verify against IM.

**3. Modal jobs & marimo notebooks** (`modal_jobs/data_collection.py`,
`notebooks/data_collection/*.py`, `notebooks/model_training/*.py`): mostly import the `src`
functions, but several contain WEIS strings in comments/paths and the backfill/rebuild
notebooks reference WEIS URLs. Sweep after `src` is done.

**4. `app.py` / `src/plotting.py`**: user-facing "WEIS" labels, links to WEIS marketplace
pages, and the settlement-location dropdown. Update copy, links, and the location universe.
*Status: label/link updates committed (`1e61c9f`); the location universe still depends on
Phase 3.*

**5. `src/parameters.py`, `src/modeling.py`**: no URL logic. New IM/West artifacts get new
names — set `MODEL_NAME='spp_west'` and new West checkpoint paths (leave the WEIS artifacts as-is; no
repo/deploy rename now — future refactor).

**6. R2 storage layout:** Reuse the **existing `spp-weis-forecast`** (no new bucket now — see
below). IM data lands in a **new, separate prefix `data_im/`** within it (`data_im/mtlf/`,
`data_im/mtrf/`, `data_im/lmp_*`, `data_im/rf_reserve_zone/`, `data_im/da_lmp/`, plus
consolidated `data_im/*.parquet`), leaving the WEIS `data/` folder untouched so both pipelines
run in parallel. Stored data keeps **both BAAs**; LMP is scoped to the hub/BA node list at
storage (amended 2026-07-05), the other feeds are stored whole with West filtering downstream.

> **Bucket naming — deferred.** Cloudflare R2 **cannot rename a bucket in place** (names are
> immutable, like S3); "renaming" means create `spp-im-bucket` → copy all objects
> (`scripts/r2_move_objects.py`) → repoint the `aws-secret` Modal secret + Posit deploy →
> delete the old bucket. That's a full copy-migration, so it's bundled into the **later rename
> refactor** (repo + Modal apps + deploy + bucket together), **not** this migration. For now the
> `weis` in the bucket name is a harmless cosmetic artifact; keeping one bucket also keeps the
> stitch step simple (WEIS `data/` + IM `data_im/` side by side, no cross-bucket reads).

---

## The hard data-science problem: retraining across the regime break

This is not just a plumbing swap. WEIS history ends 2026-04-01; RTO West history begins
2026-04-01 under a **different market design** (day-ahead market present, new resources, new
congestion patterns, new node set). Implications:

- **The current champion model is stale** — trained on WEIS prices for WEIS/PSCO nodes, which
  are being dropped. A full retrain on the new hub/BA node set is required.
- **Chosen approach (see Decisions locked): STITCH.** Each hub/BA node's WEIS history is glued
  onto its IM history into one continuous series, with a **structural-break indicator covariate
  at 2026-04-01**, trained on a **365-day window**. This gives a full-lookback model *now*
  rather than waiting until ~2027-04 for clean West-only history. The stitch is **materialized
  in storage** during the Phase 2 backfill (WEIS rows copied into the `data_im/` consolidated
  tables with `BAA='SWPW'` and a `source` column), so training just reads one continuous
  dataset — no per-retrain join.
- **Stitching verdict (verified crosswalk):** clean **only for hub/BA-level nodes**, which match
  on **exact name** — so no Pnode crosswalk is needed (the renamed nodes were all out-of-scope
  resource/load points; PSCO substations matched 0/10 but are dropped anyway).
- **Mixed-length series:** ~40 BA-level nodes get long stitched histories; the new IM hub
  constructs (`SWPW_HUB`, `CRSP_HUB`, `LAP_HUB`, `WACM_*` hubs) have no WEIS predecessor and run
  on West-only (short) history — lower-confidence until it matures. The global ensemble handles
  varying series lengths.
- Re-run the **Optuna** hyperparameter study on the stitched West data; current `TIDE_PARAMS`
  were tuned on WEIS/PSCO.
- **Day-Ahead LMP:** collected now but **not** modeled yet (deferred) — a candidate RT covariate
  and/or a future standalone DA forecasting model.

---

## Node geometry / map coordinates (lat-lon)

**Goal:** plot West hub/BA nodes on a map (and enable any geospatial features). SPP does **not**
publish settlement-location/pnode coordinates in the marketplace CSV feeds, and public GIS
layers (HIFLD control areas) are BA-polygon-only, hard to reach programmatically from here
(primary host down; reachable copies are regional clips), and need a fragile full-name→code
crosswalk. **Rejected** in favor of the authoritative source below.

**Authoritative source — SPP Price Contour Map ArcGIS service.** SPP's own price map
(`pricecontourmap.spp.org/pricecontourmap`) plots hubs, interfaces, and DC ties from a public
ArcGIS REST service keyed by **`SETTLEMENT_LOCATION`** — the exact same names as the LMP feed,
so it's a **direct join, no crosswalk**:

- Base: `https://pricecontourmap.spp.org/arcgis/rest/services/PCM/RTBM_Features/MapServer`
  (also `DA_Features`, `DELTA_Features`; `RTBM_Features` is real-time)
- Point layers: `1` DC Ties, `2` Hubs, `3` Interfaces, `4` M2M Constraints,
  `5` Binding Constraints; `6` Reserve Zones (polygons)
- Query: `/{layer}/query?where=1=1&outFields=SETTLEMENT_LOCATION,PNODETYPE,DESCRIPTION&returnGeometry=true&outSR=4326&f=json`
  (`outSR=4326` returns WGS84 lat/lon directly)

**Coverage (verified 2026-07-05):** 51 plotted points (3 hubs, 42 interfaces, 6 DC ties). For
the West hub/BA scope this gives authoritative lat/lon for:
- `SWPW_HUB` (40.65, −105.75)
- Internal West BA interfaces: `PSCO, PNM, PACE, WALC, BHBA, GRID, GWA, LAMW`
- External-seam interfaces: `AESO, AZPS, BPA, CISO, IPCO, NEVP, NWMT, PGE, SCE`
- All 6 East↔West DC ties (`…STEGALL`, `…SIDNEY`, `…MILES_CITY`)

→ **18 of 64** hub/BA nodes. (21 of the full 302 SWPW settlement locations match PCM points:
the 18 above plus the 3 West-side DC-tie endpoints `WACM.TSPM.STEGALL`, `WACM.LAP.SIDNEY`,
`WAUW.UGPW.MILES_CITY`, which sit outside the hub/BA model scope.)

**Gaps + fallback tiers.** SPP plots only a curated subset. Not covered as PCM points: the
financial/settlement hubs (`CRSP_HUB`, `LAP_HUB`, `WACM_*` interchange hubs, `LAPT.*.FSE`/
`CRSP.*.FSE`, `TSPM_SOURCEHUB`, `MEAI_CRG_HUB`, `PRPM.CRAIG1`) and several neighbor BAs
(`AVA, LADWP, SDGE, SRP, TEPC, TID, TPWR, PSEI, SCL, PACW, BANC, IID, VEA`). Resolve in order:
1. **Reserve-zone polygon centroid** (layer 6; zone 21 = West) for zone-level placement.
2. **Parent-BA coordinate** — map a settlement location to its owning BA's point by name prefix
   (e.g. `WACM_*` → `WACM`, `LAPT.*` → `LAP_HUB`).
3. **Manual** lat/lon for the handful that remain.

**Functionalize + document (maintenance utility, NOT per-run).** Geometries change rarely —
only when SPP adds/moves nodes — so this is an occasional refresh, not part of hourly
collection:
- Add `src/geometry.py` with `fetch_pcm_geometries() -> pl.DataFrame` that queries the PCM
  layers, applies the fallback tiers, and returns
  `settlement_location, lat, lon, pnode_type, source ('pcm'|'reserve_zone'|'parent_ba'|'manual')`.
- Persist to a checked-in reference file (`src/reference/node_geometry.csv`) so the app/plots
  read a static file, never the live service at request time.
- Provide a re-runnable refresh script/notebook
  (`notebooks/data_collection/refresh_node_geometry.py`) that documents the process and logs
  added/moved/removed nodes on each refresh.
- The app map reads the reference file; a node missing from it falls back to its BA centroid,
  logged.

Scoped as **Phase 3b** below — depends only on the confirmed node universe, independent of
retrain. Prototype scripts live in `scripts/node_geometry_prototype/` (`pcm_all.py` →
`west_hub_nodes_latlon.csv`, plus `west_hub_nodes.csv` and `swpw_nodes_classified.csv`) and
can seed `src/geometry.py`.

---

## Phased execution plan

**Phase 0 — Feeds & schema. ✅ DONE for the five core feeds** (verified 2026-07-05; see the
verified table above — the daily LMP rollup was confirmed via the listing API; the earlier
404s were its 5-day publication lag). The **DA LMP** slug is verified too
(`da-lmp-by-settlement-location`). Gen-capacity is dropped (dead code).

**Phase 1 — Build the parallel IM collector.** New IM collection code (alongside WEIS, writing
to `data_im/`) with new filename parsing and DST-variant handling. Keep the `BAA` column, both
BAAs; **LMP stores hub/BA node rows only** (amended 2026-07-05), MTLF/MTRF/`RF_RESERVE_ZONE`
store whole (one row per BAA/zone per interval — tiny), West filtering downstream. **Add `BAA`
to every upsert dedup key** (see touch points — clobber bug otherwise). Collectors: RTBM
5-min LMP, daily LMP (confirmed — window must end at `end_ts - 5d` per the publication lag,
see Phase 0), MTLF, MTRF,
`RF_RESERVE_ZONE` (store all zones; supplies wind/solar actuals), and DA LMP. Processors must
tolerate the missing `BAA` column in pre-launch files (fill `BAA='SPP'` — see Phase 0/Phase 2).
Unit-test each `get_*_url` and processor against a real sample CSV (one pre-launch, one
post-launch); run one collection end-to-end to `data_im/`.

**Phase 2 — Backfill history into `data_im/`.** Three segments; after this phase the
`data_im/` consolidated tables are one continuous training dataset with ≥365 days for both
BAAs, and Phase 4 needs no join/stitch logic:
- **IM era (2026-04-01 → present), both BAAs:** pulled from the IM feeds. Files carry the
  `BAA` column; LMP keeps hub/BA node rows only, same as ongoing collection.
- **Pre-launch East era (2025-04-01 → 2026-03-31):** the same feeds have years of East-only
  history (verified live 2026-07-05) — backfill one year before the seam so the East BAA also
  has ≥365 days of training data from day one (per the East-expansion rationale in Decisions).
  **LMP keeps East hub rows only** (the 10 `node_list.EAST_HUB_NODES`: the `SPPNORTH_HUB`/
  `SPPSOUTH_HUB` aggregates plus the 8 member-area trading hubs). **Pull the LMP year via the
  daily rollup** — the pre-launch `By_Day` files
  exist (verified 2026-07-05), so this era is ~365 daily pulls (~47 MB each), not ~105k
  5-min pulls. **Schema caveat:** pre-launch files have **no `BAA` column** (it
  was added at RTO West launch) — the processors must tolerate the missing column and fill
  `BAA='SPP'` (pre-launch IM was East-only; the "system-wide" MTLF/MTRF of that era are the
  East series). Verify `RF_RESERVE_ZONE` actually publishes pre-launch history — and whether
  East-era zone data is needed at all — before including it in this segment.
- **WEIS West stitch-fill (≤ 2026-03-31): materialize the stitch in storage.** One-time
  backfill script that copies the WEIS history from the existing `data/` consolidated
  parquets into the `data_im/` consolidated tables with `BAA='SWPW'` filled (WEIS was by
  definition West; its system-wide MTLF/MTRF are the West series). Copy the **hub/BA
  node-list rows** (the consolidated tables are hub-scoped — amended 2026-07-05); the ~42
  exact-name hub/BA nodes become continuous series automatically, and the stitch stays
  re-runnable from raw `data/` if scope ever widens. Because the WEIS feed is dead, this
  runs once; the hourly upsert then only ever appends IM data.
- **Provenance guardrails:** the raw `data/` (WEIS) and `data_im/` file prefixes stay
  separate and untouched — the merge happens only in the consolidated training tables, which
  carry a **`source` column (`'weis'` / `'im'`)** so every row is traceable and the stitch is
  re-runnable from raw if the seam treatment ever changes.

**Phase 3 — Data engineering & app.** Add the downstream West filters (`BAA == 'SWPW'`,
`ReserveZone == 21`); replace `proc_lmp`'s `loc_filter='PSCO_'` with the **West hub/BA node
list** (PSCO is being dropped); refresh the app's settlement-location universe, labels, and
marketplace links. Update/repair unit + e2e tests (fixtures currently assume WEIS schema;
include blank-`BAA` rows, which occur in live files).

**Phase 3b — Node geometry reference.** Build `src/geometry.py::fetch_pcm_geometries()` +
`src/reference/node_geometry.csv` + a refresh notebook, per the "Node geometry" section.
Independent of retrain; can run as soon as the node universe is fixed.

**Phase 4 — Retrain & re-tune.** The stitched series already exist in storage (Phase 2
materialized WEIS history into the `data_im/` tables), so no join logic is needed here: add
the 2026-04-01 **break-indicator covariate** — a Darts **future covariate** (it must be known
over the 120-h forecast horizon; trivially constant 1 post-seam), *not* a past covariate.
**Lifecycle:** drop it at the first retrain whose 365-day training window no longer spans the
seam (~2027-04), when it degenerates to a constant. Set `MODEL_NAME='spp_west'`; re-run
Optuna; evaluate against a West holdout; promote a new champion.

**Phase 5 — Deploy, decommission WEIS jobs, docs.** Deploy the IM Modal jobs and confirm they
run on schedule; **then remove/undeploy the WEIS Modal collection jobs** (`collect_hourly`,
`collect_daily`) — the WEIS feed is dead so they collect nothing. **Keep** the WEIS historical
R2 data (`data/`) for stitching. Update `README.md` and sweep IM notebooks. (Full rename —
repo, Modal apps, Posit deploy, **and R2 bucket `spp-weis-forecast`→`spp-im-bucket`** via
copy-migration — is a **later refactor**, not now.)

> **Partially done early (2026-07-05):** the WEIS `spp-weis-data-collection` Modal app is
> **stopped**, and the six WEIS market-collection notebooks + the WEIS Modal wrapper were
> moved to **`deprecated/weis/`** (`git mv`, history preserved).
>
> **Helper extraction done (2026-07-08):** the shared, feed-agnostic collection helpers
> (`get_csv_from_url`, `_s3_storage_options`, `set_he`, `ProgressParallel`, …) now live in
> **`src/data_collection_utils.py`**, their single home. `data_collection_im.py` (live IM
> collector) and `scripts/weis_stitch_fill.py` import from it, so neither depends on the WEIS
> module. `data_collection.py` re-imports the same helpers for its WEIS feed logic and **stays
> in `src/`** for now. Its only remaining caller,
> `notebooks/data_collection/data_collection_weather.py` (`data_collection.upsert_weather`), is
> itself unwired — weather is **not** an active covariate (the `prep_weather` join and the
> `temperature` column are commented out in `data_engineering.py`). Fully retiring the WEIS
> module to `deprecated/` is therefore a self-contained follow-up (module + weather notebook +
> `test_data_collection.py`), not blocked by anything live.

**Suggested sequencing:** all work on the **feature branch**. Phase 1 → 2 → 3/3b restore +
enrich the data pipeline and can proceed now. Phase 4 (retrain) follows once the
backfill and node-geometry reference land. Merge to `main` after the IM pipeline is validated;
Phase 5 decommissions the WEIS jobs.

---

## Decisions locked (interview 2026-07-05)

**Infrastructure & scope**
- **Storage scope (amended in the 2026-07-05 plan-review interview):** both BAAs, keeping the
  `BAA` column — but **LMP stores hub/BA node rows only** (the West hub/BA/seam list + the
  East hubs), not the full location universes. MTLF/MTRF and `RF_RESERVE_ZONE` are stored
  whole (all BAAs / all reserve zones — one row per BAA/zone per interval, so tiny);
  `BAA == 'SWPW'` / `ReserveZone == 21` remain downstream filters. Rationale for the LMP
  narrowing: a year of full-universe East 5-min LMP (thousands of locations vs WEIS's ~348)
  is a large storage/consolidation footprint with no current use, and the portal keeps years
  of history — widening scope later is a re-backfill, not data loss. The hub/BA node list
  gets **one home** (`src/parameters.py` or `src/reference/`) shared by collection and
  engineering.
- **Upsert dedup keys gain `BAA`** (2026-07-05 review): mtlf/mtrf currently key on
  `GMTIntervalEnd` alone, so two stored BAAs would silently clobber each other. New keys:
  mtlf/mtrf `(GMTIntervalEnd, BAA)`; lmp adds `BAA` to `(GMTIntervalEnd_HE,
  Settlement_Location_Name, PNODE_Name)`; `rf_reserve_zone` `(GMTIntervalEnd, BAA,
  ReserveZone)`.
- **Daily LMP: RESOLVED — keep the collector** (2026-07-05): the listing-API search found the
  feed at the WEIS-analogous `By_Day` path; the earlier 404s were its **5-day publication
  lag**. Keep the daily repair-sweep collector with a lag-aware window (days ending at
  `end_ts - 5d`), and use the pre-launch daily files for the Phase 2 East LMP backfill.
- **East history too:** the Phase 2 backfill reaches back to **2025-04-01** (one year before
  the seam) so the East BAA also starts with ≥365 days of training data — pre-launch files are
  East-only and lack the `BAA` column; processors fill `BAA='SPP'`.
- **Model scope:** **West hub/BA-level nodes** — both the SWPW-internal hubs/BAs **and** the
  ~25 external-seam neighbor interfaces (`CISO`, `BPA`, `AESO`, `AZPS`…) as additional forecast
  series (global model benefits + seam prices drive West prices). `SWPW_HUB` is the flagship
  target. **PSCO substation focus is dropped** — replace `loc_filter='PSCO_'` with the hub/BA
  node list.
- **Feature branch:** do **all** of this work on a dedicated feature branch (e.g.
  `feature/rto-west-migration`), not `main`. Merge only once the IM pipeline is validated.
- **Parallel pipeline → then decommission WEIS jobs:** build the IM collector **alongside** WEIS
  (separate R2 prefix `data_im/`, WEIS code untouched) so nothing breaks during the build.
  Because the **WEIS feed is dead** (no new data since 2026-04-01), once the IM collection jobs
  are deployed and confirmed running, **remove/undeploy the WEIS Modal collection jobs** — they
  collect nothing. **Keep** the WEIS historical R2 data (`data/`); it's needed for stitching.

**Timeline & training**
- **Timeline:** build-it-right migration, **no hard date, open-ended**. No external pressure →
  no throwaway interim model needed.
- **Training-data strategy: STITCH** each hub/BA node's WEIS history onto its IM history into one
  continuous series, with a **structural-break indicator covariate at 2026-04-01** (a Darts
  *future* covariate, dropped once the training window is fully post-seam — see Phase 4). Use
  a **1-year (365-day) training window** (`TRAIN_START='365D'`, unchanged) — stitching makes
  a full lookback achievable now instead of waiting until ~2027-04.
  - **Materialized in storage, not joined at train time:** the Phase 2 backfill copies WEIS
    history into the `data_im/` consolidated tables (`BAA='SWPW'`, `source='weis'`), copying
    the hub/BA node-list rows (the tables are hub-scoped — amended 2026-07-05). Raw `data/`
    and `data_im/` file prefixes stay separate for provenance; the stitch is re-runnable from
    raw if scope widens.
  - **No Pnode crosswalk needed:** hub/BA nodes stitch on **exact name**; the renamed nodes were
    all out-of-scope resource/load points.
  - **Mixed-length caveat:** stitching only helps the ~40 BA-level nodes with WEIS predecessors.
    New IM hub constructs (`SWPW_HUB`, `CRSP_HUB`, `LAP_HUB`, `WACM_*` hubs) have **no WEIS
    history** → West-only (short) series; their forecasts are lower-confidence until history
    matures. The global ensemble handles mixed-length series fine.
- Re-run the **Optuna** study on the stitched West data once the pipeline lands; current
  `TIDE_PARAMS` were tuned on WEIS/PSCO.

**Day-Ahead market**
- **Collect DA LMP now, DEFER modeling it.** Add a DA LMP collector to the IM pipeline (both
  BAAs, `data_im/`) so history accrues, but do **not** wire it into the RT model yet. Possible
  future work: a **standalone DA price-forecasting model**.

**Naming (option a — new names for new things; full rename deferred)**
- New IM/West artifacts get clear names: `MODEL_NAME='spp_west'`, R2 prefix `data_im/`
  **inside the existing `spp-weis-forecast`**, Modal `spp-im-*` apps. **Do not** rename the repo,
  the running WEIS pipeline, the R2 bucket, or the Posit Connect deploy now.
- **Bucket stays `spp-weis-forecast`:** R2 can't rename in place, so moving to `spp-im-bucket`
  is a create-new + copy-migrate + secret/deploy repoint — bundled into the **future rename
  refactor** (repo + Modal + deploy + bucket together), not this migration.

## Open questions

All **decisions** are resolved (2026-07-05 interviews above). One **verification item**
remains open:

1. **East-era `RF_RESERVE_ZONE` need** — the feed's pre-launch availability is verified
   (real 2025-07-01 file in `tests/unit/fixtures/`), but decide whether East-era zone data
   is needed at all (the actuals matter for the West model, not East).

~~DA LMP slug/schema~~ — **RESOLVED**: slug is `da-lmp-by-settlement-location`; schema
handled in `src/data_collection_im.py` (mixed timestamp formats).

~~East hub settlement-location names~~ — **RESOLVED 2026-07-05**: the East scope is the 10
hub-level nodes in `node_list.EAST_HUB_NODES` — the `SPPNORTH_HUB`/`SPPSOUTH_HUB` aggregates
plus the 8 member-area trading hubs (`CSWS_HUB`, `ETEC_HUB`, `GRDA_HUB`, `GSEC_HUB`,
`HAST_TNSK_HUB`, `KCPL_GMOC_HUB`, `LES_HUB`, `SECI_HUB`), all confirmed present in a live
post-launch LMP file. Scoped at the hub level, not by reserve zone: SPP publishes **no
node↔reserve-zone crosswalk** in the file-browser feeds (the `RF_RESERVE_ZONE` feed labels
East zones as bare integers 1–5 with no node membership; likely reference slugs 404).

~~Daily LMP rollup feed~~ — **RESOLVED 2026-07-05**: exists at the WEIS-analogous `By_Day`
path with a 5-day publication lag (see Phase 0); collector kept, lag-aware window.
