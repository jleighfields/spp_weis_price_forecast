"""Single home for the stored/modeled settlement-location node lists.

The RTO West migration scopes LMP storage and modeling to hub/BA-level
nodes (plans/weis_to_rto_west_migration.md, decisions 2026-07-05). This
module is that list's one home: the IM collectors filter LMP rows to
STORED_NODES at storage time, and data engineering derives the model
universe from the same lists — neither redeclares them.

Kept separate from src/parameters.py on purpose: parameters.py imports
sklearn/darts, which the Modal data-collection image does not install.

Provenance: WEST_HUB_BA_NODES comes from the classified SWPW universe in
scripts/node_geometry_prototype/west_hub_nodes.csv; all 64 names were
verified present (exact match) in a live post-launch LMP file on
2026-07-05, as were the two East trading hubs.
"""

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

# SPP East BAA trading hubs (the only East series stored; verified names).
EAST_HUB_NODES = ['SPPNORTH_HUB', 'SPPSOUTH_HUB']

# What the IM collectors keep from LMP files (both BAAs).
STORED_NODES = WEST_HUB_BA_NODES + EAST_HUB_NODES
