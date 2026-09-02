"""Shared test factory.

Ported from smDA-Python's tests/helpers.py, reduced to the one factory this
package needs: the others built loader, batch and SMLM configs that have no
counterpart here.

The values are deliberately NOT the bundled dataset's.  A test that happened
to pass because the factory supplied the right numbers would not be testing
anything, so tests that care about the bundled data state its parameters
themselves.
"""
from __future__ import annotations

import os
from pathlib import Path

from smda_hmm.vbhmm.model import VBHMMParams


def v4_sample_dir() -> Path | None:
    """Directory holding an AAS v4 pair, or None if none was supplied.

    The deposited data is v2 throughout, so this repository contains no v4
    file to test the v4 half of the reader and writer against.  Writing one
    would mean inventing the columns v2 does not carry, into a file that
    reads as measured data, so instead the v4 tests read a real pair from a
    checkout that has one -- smDA-Python's ``data/Sample`` -- named by the
    environment variable ``SMDA_V4_SAMPLE``::

        set SMDA_V4_SAMPLE=C:/Users/yanag/smda-python/data/Sample

    Unset, the v4 tests skip and say so.  Set but wrong, this raises: a
    misspelt path that quietly went back to skipping would look exactly like
    a clean run.  The v4 tests were doing precisely that until 2026-09-02 --
    they were gated on ``data/_absent_v4``, a path that cannot exist, with no
    way to point them at anything real.
    """
    raw = os.environ.get("SMDA_V4_SAMPLE")
    if not raw:
        return None
    d = Path(raw)
    missing = [n for n in ("Sample_data.csv", "Sample_hmm.csv")
               if not (d / n).is_file()]
    if missing:
        raise RuntimeError(
            f"SMDA_V4_SAMPLE={raw} does not hold {', '.join(missing)}")
    return d


def make_test_vbhmm_params(**overrides) -> VBHMMParams:
    """VBHMMParams with the two measurement conditions filled in.

    VBHMMParams refuses to be constructed without timestep and
    distance_per_pixel, which is the point; tests still have to supply
    something, so this supplies an arbitrary valid pair and lets each test
    override whatever it is actually about.
    """
    defaults = dict(timestep=0.0333, distance_per_pixel=0.066)
    defaults.update(overrides)
    return VBHMMParams(**defaults)
