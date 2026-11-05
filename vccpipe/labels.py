"""Display names and lookup tables for the notebook's tables and figures.

Used in: sections 4 to 10. Nothing here changes a computation except `HALF_LINE_TISSUE`, which groups the six
split-half lines into tissues for section 6's within/across comparison.
"""

__all__ = [
    "FOLD_LABEL",
    "HALF_LINE_LABEL",
    "HALF_LINE_TABLE",
    "HALF_LINE_TISSUE",
    "METRIC_LABEL",
    "SCREEN_LABEL",
    "TYPE_LABEL",
    "TYPE_TISSUE",
]

# Section 4: each public screen's display name, by its table name.
SCREEN_LABEL = {
    "orion_hct116": "HCT116 (Orion)",
    "orion_hek293t": "HEK293T (Orion)",
    "k562gwps_bulk": "K562 genome-wide",
    "bench_k562": "K562 essential",
    "viperturb": "K562 (Flex)",
    "bench_rpe1": "RPE1 essential",
    "bench_hepg2": "HepG2 essential",
    "bench_jurkat": "Jurkat essential",
    "bench_h1": "H1 (2025 train)",
    "arc_h1": "H1 (2025 held-out)",
    "hesc_gs": "H1 genome-scale",
    "kolf": "KOLF iPSC",
    "cd4_Rest": "CD4 T rest",
    "cd4_Stim8hr": "CD4 T 8 h",
    "cd4_Stim48hr": "CD4 T 48 h",
    "dld1_gx": "DLD-1",
    "a549_ko": "A549 knockout",
    "a549_myllia": "A549 CRISPRi",
    "a375_perturbfate": "A375",
}

# Sections 5 and 6: the six lines whose cells were split into halves, by their halves file; the source table of each;
# and the tissue section 6 groups them by.
HALF_LINE_LABEL = {
    "bench_k562": "K562",
    "bench_jurkat": "Jurkat",
    "bench_hepg2": "HepG2",
    "bench_rpe1": "RPE1",
    "h1train": "H1 (Flex)",
    "kolf": "KOLF iPSC",
}
HALF_LINE_TABLE = {
    "bench_k562": "bench_k562",
    "bench_jurkat": "bench_jurkat",
    "bench_hepg2": "bench_hepg2",
    "bench_rpe1": "bench_rpe1",
    "h1train": "bench_h1",
    "kolf": "kolf",
}
HALF_LINE_TISSUE = {
    "bench_k562": "blood",
    "bench_jurkat": "blood",
    "bench_hepg2": "liver",
    "bench_rpe1": "retinal epithelium",
    "h1train": "pluripotent",
    "kolf": "pluripotent",
}

# Section 8: each held-out training line (a fold), and each cell type of the training screens, with its tissue.
FOLD_LABEL = {
    "h1": "H1 stem cells",
    "k562gwps": "K562 genome-wide",
    "k562": "K562 essential",
    "jurkat": "Jurkat",
    "rpe1": "RPE1",
    "hepg2": "HepG2",
}
TYPE_LABEL = {
    "h1": "H1",
    "ipsc": "KOLF iPSC",
    "k562": "K562",
    "jurkat": "Jurkat",
    "cd4t": "CD4 T",
    "hepg2": "HepG2",
    "rpe1": "RPE1",
    "hct116": "HCT116",
    "hek293t": "HEK293T",
    "dld1": "DLD-1",
    "a549": "A549",
    "a375": "A375",
    "pathway": "pathway\nscreens",
}
TYPE_TISSUE = {
    "h1": "pluripotent",
    "ipsc": "pluripotent",
    "k562": "blood",
    "jurkat": "blood",
    "cd4t": "blood",
    "hepg2": "liver",
    "rpe1": "epithelial",
    "hct116": "epithelial",
    "hek293t": "epithelial",
    "dld1": "epithelial",
    "a549": "epithelial",
    "a375": "other",
    "pathway": "other",
}

# Section 10: what each of the scorer's six measures reads.
METRIC_LABEL = {
    "pds": "PDS: which knockdown is which",
    "mse": "MSE: pooled profile error",
    "nmae": "NMAE: size of the called changes",
    "fid": "FID: direction of the called genes",
    "reach": "REACH: called genes reached",
    "jac": "JAC: the called set",
}
