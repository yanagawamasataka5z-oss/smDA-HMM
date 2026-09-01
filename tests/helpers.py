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

from smda_hmm.vbhmm.model import VBHMMParams


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
