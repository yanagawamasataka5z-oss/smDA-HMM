"""Regression tests for the dead-parameter fixes (2026-08-31).

Background
----------
`is_add_each_trajectory` had a UI checkbox, was read from settings.csv, stored
in VBHMMParams, saved with the settings and written into the hmm.csv metadata —
and still had no effect, because it was never passed to the Rust binding, which
is the production path.  Rust always scaled priors by the trajectory count.
Reading the code did not reveal this; only running with two different values
did.  See scripts/b1_dead_parameter_audit.py and lessons.md.

These tests pin the three things that were wrong.
"""

import numpy as np
import pytest

from smda_hmm.vbhmm.model import SPATIAL_DIM, VBHMMParams, _build_priors

smda_scan = pytest.importorskip("smda_scan")


K = 2
DX2 = np.array([0.0010, 0.0021, 0.0032, 0.0043, 0.0015, 0.0026, 0.0037,
                0.0048], dtype=np.float64)
STARTS = np.array([0, 4], dtype=np.int64)
ENDS = np.array([3, 7], dtype=np.int64)


def _run(**over):
    kw = dict(min_hidden=K, max_hidden=K, w_pi_scale_mode=1,
              w_b_scale_mode=1, seed=42)
    kw.update(over)
    return smda_scan.run_vbhmm_model_selection(DX2, STARTS, ENDS, **kw)


def _vec(res, key="state_n"):
    return np.asarray(res["models"][0][key], dtype=float)


# ---------------------------------------------------------------------------
# 1. is_add_each_trajectory must reach the Rust engine
# ---------------------------------------------------------------------------

class TestAddPerTrajectoryReachesRust:
    """Before the fix, Rust ignored this flag and always scaled by M.

    NOTE ON WHAT IS ASSERTED HERE
    -----------------------------
    This class fixes only that OFF produces a *different* result from ON.  It
    deliberately does NOT assert that Rust OFF equals Python OFF.

    Why the weaker condition: asserting agreement requires a tolerance, and no
    defensible tolerance exists yet.  Rust and Python still disagree with each
    other on the ON path too (measured: D 5.7-82 ppm on the validation cells,
    and ~4x worse on 260830SampleData/t000028/t0002, cause not identified).  A
    tolerance chosen today would simply bake that unexplained implementation
    difference into the acceptance criterion.

    Planned upgrade: once the Rust/Python difference is diagnosed (carry-over
    items 1 and 2 in todo.md), replace this with a real agreement test —
    `assert Rust OFF == Python OFF` within the tolerance established then.
    Until that happens, "OFF differs from ON" is the honest assertion: it is
    exactly what the fix changed, and nothing more.
    """

    def test_off_differs_from_on(self):
        on = _vec(_run(is_add_each_trajectory=True))
        off = _vec(_run(is_add_each_trajectory=False))
        assert not np.array_equal(on, off), (
            "is_add_each_trajectory has no effect on the Rust result — the "
            "flag is not reaching the engine again (this is the exact defect "
            "fixed on 2026-08-31)."
        )

    def test_default_is_on(self):
        """AAS uses ON, and every past validation ran ON; the default must
        preserve that so existing results stay reproducible."""
        default = _vec(_run())
        on = _vec(_run(is_add_each_trajectory=True))
        assert np.array_equal(default, on)

    def test_on_scales_priors_by_trajectory_count(self):
        """Sanity check on direction: ON must give the larger prior n."""
        on = _vec(_run(is_add_each_trajectory=True))
        off = _vec(_run(is_add_each_trajectory=False))
        assert on.sum() > off.sum()

    def test_python_priors_honour_the_flag(self):
        p_on = VBHMMParams(timestep=0.04, distance_per_pixel=0.067,
                           is_add_each_trajectory=True)
        p_off = VBHMMParams(timestep=0.04, distance_per_pixel=0.067,
                            is_add_each_trajectory=False)
        assert _build_priors(K, 100, p_on).n[0] == 100 * p_on.n_tilde
        assert _build_priors(K, 100, p_off).n[0] == p_off.n_tilde


# ---------------------------------------------------------------------------
# 2. Unimplemented scale modes must raise, not fall through
# ---------------------------------------------------------------------------

class TestScaleModeValidation:
    """S69 recorded the cost of the silent fallback: a mis-set mode produced a
    flipped BestN and a 23% D error (session69_handoff.md section 4)."""

    @pytest.mark.parametrize("mode", [2, 3, 255])
    def test_rust_rejects_unimplemented_w_b_mode(self, mode):
        with pytest.raises(ValueError, match="w_b_scale_mode"):
            _run(w_b_scale_mode=mode)

    @pytest.mark.parametrize("mode", [2, 3, 255])
    def test_rust_rejects_unimplemented_w_pi_mode(self, mode):
        with pytest.raises(ValueError, match="w_pi_scale_mode"):
            _run(w_pi_scale_mode=mode)

    def test_rust_error_mentions_the_s69_renumbering(self):
        """Old mode 2 became the current mode 1; the message must say so or
        the reader will assume mode 2 was silently accepted before."""
        with pytest.raises(ValueError, match="S69"):
            _run(w_b_scale_mode=2)

    @pytest.mark.parametrize("mode", [2, 3])
    def test_python_rejects_unimplemented_modes(self, mode):
        for field in ("w_pi_scale_mode", "w_b_scale_mode"):
            p = VBHMMParams(timestep=0.04, distance_per_pixel=0.067,
                            **{field: mode})
            with pytest.raises(ValueError, match=field):
                _build_priors(K, 10, p)

    @pytest.mark.parametrize("mode", [0, 1])
    def test_implemented_modes_are_accepted(self, mode):
        _run(w_b_scale_mode=mode)
        _run(w_pi_scale_mode=mode)


# ---------------------------------------------------------------------------
# 3. dim is not a parameter
# ---------------------------------------------------------------------------

class TestSpatialDimIsFixed:
    """preprocess_trajectories collapses the data to dx2 = dx^2 + dy^2, so the
    model is 2D by construction.  A dim knob would let a caller request 3D and
    get a silently wrong answer — worse than a dead parameter.  AAS carries no
    dimension field either (settings.csv is headed "Tracking2DSettings")."""

    def test_constant_is_two(self):
        assert SPATIAL_DIM == 2

    def test_params_has_no_dim_field(self):
        p = VBHMMParams(timestep=0.04, distance_per_pixel=0.067)
        assert not hasattr(p, "dim"), (
            "dim was reintroduced as a parameter; it must stay a module "
            "constant so 3D cannot be requested."
        )

    def test_dim_cannot_be_passed_to_rust(self):
        with pytest.raises(TypeError):
            _run(dim=3)
