"""The final recipe, pinned: every number and list the submitted file was built with.

Each value is fingerprinted (sha256 of its JSON form); the fingerprints were taken from the code that built the
submitted file. A failure here means a change to the recipe itself, which would change the file.
"""

import hashlib
import json

import numpy as np
import pytest

import vccpipe

FINGERPRINTS = {
    "TS": "c953d46fad8b859d",
    "CLIP": "8ab31b5afaea5611",
    "SE_INFLATE": "004a9e0878ff83e6",
    "N_CTRL_DEFAULT": "39e5b4830d4d9c14",
    "DEPTH": "75bea235492015f5",
    "TYPE_W": "3bcefd6482ce68ab",
    "FLEX_TYPES": "adc9f579e3610878",
    "CIS_EDGES": "19c9a31d00b8523b",
    "CIS_CPM_EDGES": "be227a3db5e270f8",
    "CELL_L_EDGES": "f239ad4bdd46aaa6",
    "CELL_E_EDGES": "921108ae773cf33a",
    "CELL_TABLE": "cef09be7eac4cbda",
    "CIS_PRIOR": "ed8ec9061cff1135",
    "E_TRAIN_EXTRA": "3a0e41cbdc4cab3d",
    "E_ROW_EXTRA": "ee632c2d9ebfa4b4",
    "POWER_SHIFTS": "a25838c38bed3f83",
    "TYPES": "27053e0efe2df72a",
    "TARGETS": "0d0a7bc1977971c2",
    "SIB": "81f6b8111d943529",
    "P53_CORE": "acc1eaaa8986f092",
    "P53_SOURCES": "478873917e312386",
    "P53_KEY2": "4020836c0b5ff03e",
    "PIECES": "bc305979ec5ae334",
    "FINAL_SOURCES": "c36a4263a679de17",
    "PROBE_SOURCES": "ddbebcfe1532ae53",
    "NO_ORION_SOURCES": "02c2bfb806c3d097",
    "SOURCE_TYPE": "b34fac8bc3345d68",
    "FINAL_CONFIG": "245a8b24beaeb021",
    "THREADS": "bb96ecacca90d33e",
    "CELLS_PER_PERT": "26d228663f13a885",
}
# Every public download, its size and sha256 (as the appendix and FAST read them).
DOWNLOADS = (197, "b37a6733d91263a6")


def plain(x):
    """A value as plain JSON types, with dict keys and sets sorted."""
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, dict):
        return {str(k): plain(x[k]) for k in sorted(x)}
    if isinstance(x, set | frozenset):
        return sorted(plain(i) for i in x)
    if isinstance(x, list | tuple):
        return [plain(i) for i in x]
    if isinstance(x, np.floating):
        return float(x)
    return x


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(plain(value)).encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(FINGERPRINTS))
def test_recipe_value_is_unchanged(name):
    assert fingerprint(getattr(vccpipe, name)) == FINGERPRINTS[name]


def test_download_list_is_unchanged():
    files = vccpipe.PUBLIC_FILES
    assert (len(files), hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]) == DOWNLOADS


def test_recipe_is_consistent():
    assert set(vccpipe.WATERFALL) == set(vccpipe.PIECES)  # the waterfall switches exactly the final recipe's pieces
    assert set(vccpipe.FINAL_SOURCES) <= set(vccpipe.SOURCE_TYPE)
    assert len(vccpipe.E_TRAIN_EXTRA) == 145 and vccpipe.E_TRAIN_EXTRA == sorted(set(vccpipe.E_TRAIN_EXTRA))
    assert len(vccpipe.E_ROW_EXTRA) == 853 and vccpipe.E_ROW_EXTRA == sorted(set(vccpipe.E_ROW_EXTRA))
    assert np.asarray(vccpipe.CELL_TABLE).shape == (len(vccpipe.CELL_L_EDGES) + 1, len(vccpipe.CELL_E_EDGES) + 1)
    assert np.asarray(vccpipe.CIS_PRIOR["table"]).shape == (len(vccpipe.CIS_EDGES) - 1, 4)
