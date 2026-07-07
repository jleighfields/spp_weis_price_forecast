"""Single home for the stored/modeled settlement-location node lists.

The RTO West migration scopes LMP storage and modeling to hub/BA-level
nodes. This module is that list's one home: the IM collectors filter LMP rows to
STORED_NODES at storage time, and data engineering derives the model
universe from the same lists — neither redeclares them.

Kept separate from src/parameters.py on purpose: parameters.py imports
sklearn/darts, which the Modal data-collection image does not install.
This module holds only pandas-level West-scoping constants, so both the
collectors and the modeling pipeline can import it.

Provenance: WEST_HUB_BA_NODES comes from the classified SWPW universe in
scripts/node_geometry_prototype/west_hub_nodes.csv; all 64 names were
verified present (exact match) in a live post-launch LMP file on
2026-07-05, as were the two East trading hubs.
"""

import pandas as pd

# RTO West go-live / WEIS→IM seam. Files carry the BAA column from this
# date on, the West BAA's own market data starts here, and modeling uses it
# for the break-indicator covariate. Single home; data_collection_im and
# data_engineering both read it (data_collection_im re-exports it).
RTO_WEST_LAUNCH = pd.Timestamp('2026-04-01')

# SPP West BAA (SWPW) internal hubs and BA-level nodes, plus the
# East<->West seam constructs (SWPW_HUB is the flagship forecast target).
WEST_INTERNAL_NODES = [
    'AVRN', 'BHBA', 'CEN', 'CHPD',
    'CRSP.CSU.FSE', 'CRSP.LAPT.FSE', 'CRSP.PRPA.FSE', 'CRSP.TSGT.FSE',
    'CRSP_HUB', 'DEAA', 'DOPD', 'EPE', 'GCPD', 'GRID', 'GWA', 'LAMW',
    'LAPT.CSU.FSE', 'LAPT.PRPA.FSE', 'LAPT.TSGT.FSE', 'LAPT.UGP.FSE',
    'LAP_HUB', 'MEAI_CRG_HUB', 'MEAN.CPW.HUB24', 'PACE', 'PNM',
    'PRPM.CRAIG1', 'PSCO', 'RCDC', 'SCSW', 'SWPW_HUB', 'TSPM_SOURCEHUB',
    'WACM.PSC2.CRAIG1', 'WACM_CRSP_WILW', 'WACM_CRSP_WMPA',
    'WACM_NTUA_AZPS', 'WACM_NTUA_PNM', 'WACM_PACE_NCODY', 'WALC',
    'WAUW.NTWK',
]

# Neighbor-BA interfaces on the Western Interconnection seam. Seam prices
# drive West prices, so these are forecast series too (global-model input).
WEST_SEAM_NODES = [
    'AESO', 'AVA', 'AZPS', 'BANC', 'BCHA', 'BPA', 'CISO', 'IID', 'IPCO',
    'LADWP', 'NEVP', 'NWMT', 'PACW', 'PGAE', 'PGE', 'PSEI', 'SCE', 'SCL',
    'SDGE', 'SRP', 'TEPC', 'TID', 'TPWR', 'VEA', 'WWA',
]

WEST_HUB_BA_NODES = WEST_INTERNAL_NODES + WEST_SEAM_NODES

# The SPP West balancing authority area code (the modeled BAA). Single home
# for the value; the data-engineering West filters read it.
WEST_BAA = 'SWPW'

# SPP East BAA trading hubs (the only East series stored). The two
# SPPNORTH/SPPSOUTH aggregates plus the eight member-area trading hubs;
# all verified present (exact match) in a live post-launch LMP file on
# 2026-07-05. SPP publishes no node<->reserve-zone crosswalk, so the East
# is scoped at the hub level rather than by reserve zone.
EAST_HUB_NODES = [
    'SPPNORTH_HUB', 'SPPSOUTH_HUB',
    'CSWS_HUB', 'ETEC_HUB', 'GRDA_HUB', 'GSEC_HUB',
    'HAST_TNSK_HUB', 'KCPL_GMOC_HUB', 'LES_HUB', 'SECI_HUB',
]

# What the IM collectors keep from LMP files (both BAAs).
STORED_NODES = WEST_HUB_BA_NODES + EAST_HUB_NODES

# The West nodes actually modeled and offered in the app — a curated subset
# of STORED_NODES chosen for history coverage (every node here has ~365 days,
# either an exact WEIS match or a WACM/PSCM/BHCE/CRSP stitch proxy). Storage
# stays broad (STORED_NODES); modeling reads this. prep_lmp defaults to it.
MODEL_APP_NODES = [
    # internal West (EPE/PACE/PNM/WALC exact-match; PSCO/BHBA/SWPW_HUB/
    # WACM_CRSP_WILW proxied via the WEIS stitch)
    'SWPW_HUB', 'PSCO', 'BHBA', 'WACM_CRSP_WILW',
    'EPE', 'PACE', 'PNM', 'WALC',
    # external-seam representatives: the 25 WECC interfaces are near-identical
    # (one CAISO/WECC seam signal), so keep just two — Northwest and California.
    'BPA', 'CISO',
]
