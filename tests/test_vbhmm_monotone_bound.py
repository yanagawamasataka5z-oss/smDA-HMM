"""The VBEM convergence bound must be monotone non-decreasing.

Why this test exists
--------------------
Until 2026-08-31 the convergence test was keyed on the REPORTED bound F, whose
KL is measured against the unscaled hyperparameters while the M-step fits with
priors scaled by the trajectory count M.  F is therefore the bound of a
different model than the one being fitted, the VBEM monotonicity theorem does
not cover it, and measurement showed it rising, peaking and then descending to
the fixed point.  `|dF|/F < 1e-8` fired at that turning point: on
260830SampleData/t000028/t0002 it stopped at iteration 29 with D still 2864 ppm
from the fixed point, which is the whole of that cell's 2411 ppm disagreement
with AAS.

Two corrections were needed together — evaluate lnZ and the KL at the same
state, AND measure the KL against the same scaled priors the M-step uses.
Either alone leaves the sequence non-monotone, which is why both hypotheses
were refuted when tested separately (scripts/b1_monotonicity_matrix.py).

Monotonicity is a structural property of VBEM, not a numerical accident, so a
future change that breaks it is a defect regardless of how close the answer
looks.  This pins it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from smda_hmm.vbhmm.model import (SPATIAL_DIM, VBEM_CONV_TOL, VBHMMParams,
                             _build_priors, _compute_lower_bound, _e_step,
                             _m_step, kmeans_init, preprocess_trajectories,
                             run_vbhmm)
from smda_hmm.io.aas_reader import load_aas_settings_csv

REPO = Path(__file__).resolve().parents[1]
RTK_SETTINGS = REPO / "data" / "sample" / "settings.csv"

# A decrease of this relative size is floating-point noise, not a real one.
# Measured worst case at the fixed point: 1.5e-10 on a bound of 5e4, i.e.
# 3e-15 relative.  1e-13 leaves two orders of headroom and is still four
# orders below the -11.6 nat (2e-4 relative) drop the old bound showed.
NOISE_FLOOR = 1e-13


def _params() -> VBHMMParams:
    s = load_aas_settings_csv(RTK_SETTINGS)
    return VBHMMParams(
        n_tilde=s["n_tilde"], c_tilde=s["c_tilde"], w_pi_tilde=s["w_pi_tilde"],
        w_b_tilde=s["w_b_tilde"], mag=s["mag"], min_hidden=s["min_hidden"],
        max_hidden=s["max_hidden"], max_iter=100, num_run=1,
        frame_minimum=s["vbhmm_min_frame"], estimate_mode=s["estimate_mode"],
        is_add_each_trajectory=s["add_per_traj"],
        is_calc_kl_each=s["calc_kl_each"],
        timestep=0.040, distance_per_pixel=0.067)


def _cells() -> list[tuple[str, Path]]:
    """Every cell with real data, both validation sets."""
    out = []
    root = REPO / "data" / "sample"
    if root.is_dir():
        for c in sorted(root.glob("*/*33fps.csv")):
            if not c.name.endswith("_hmm.csv"):
                out.append((c.parent.name.replace("egfr-EGF_", "") + "/"
                            + c.name.split("_")[3], c))
    for sub in ("EGFR", "ERBB3", "ERBB4"):
        d = REPO / "data" / "RTK" / sub
        if not d.is_dir():
            continue
        for c in sorted(d.glob("*33fps.csv")):
            if c.name.endswith("_hmm.csv") or "trackmate" in c.name:
                continue
            if Path(str(c).replace(".csv", "_hmm.csv")).exists():
                p = c.name.split("_")
                out.append((f"{sub}/{p[1]}_{p[2]}", c))
    return out


CELLS = _cells()
pytestmark = pytest.mark.skipif(
    not RTK_SETTINGS.exists() or not CELLS,
    reason="validation data not present in this checkout")


def _bound_trace(csv: Path, k: int, iters: int) -> tuple[list[float], int | None]:
    """Convergence bound per iteration, plus the iteration the rule fires."""
    p = _params()
    d = preprocess_trajectories(str(csv), p)
    priors = _build_priors(k, d.n_trajectories, p)
    state = kmeans_init(d, k, priors, p)
    seq: list[float] = []
    fired = None
    for it in range(1, iters + 1):
        pst, wA, lnZz, lnZQ, lnZq = _e_step(
            d.dx2, state, d.trj_starts, d.trj_ends, SPATIAL_DIM)
        seq.append(_compute_lower_bound(
            lnZQ, lnZq, lnZz, state, priors, p.mag)[0])
        old_n, old_c = state.n.copy(), state.c.copy()
        state = _m_step(d.dx2, pst, wA, priors, d.trj_starts, SPATIAL_DIM)
        if fired is None and len(seq) > 1:
            rel = abs(seq[-1] - seq[-2]) / max(abs(seq[-1]), 1e-300)
            par = max(np.max(np.abs(state.n - old_n) / np.abs(old_n)),
                      np.max(np.abs(state.c - old_c) / np.abs(old_c)))
            if rel < VBEM_CONV_TOL and par < 1e-2:
                fired = it
    return seq, fired


@pytest.mark.parametrize("tag,csv", CELLS, ids=[t for t, _ in CELLS])
def test_convergence_bound_is_monotone(tag, csv):
    seq, _ = _bound_trace(csv, 3, 120)
    rel = np.diff(seq) / np.abs(seq[1:])
    i = int(np.argmin(rel))
    assert rel[i] > -NOISE_FLOOR, (
        f"{tag}: the convergence bound decreased by {rel[i]:.3e} (relative) at "
        f"iteration {i + 2}.  VBEM guarantees this sequence is non-decreasing, "
        f"so a real decrease means lnZ and the KL are no longer being "
        f"evaluated on the same model — check that run_vbhmm still passes the "
        f"M-scaled `priors` (not `kl_priors`) to the convergence bound and "
        f"evaluates it before the M-step.")


@pytest.mark.parametrize("k", [1, 2, 4, 5])
def test_monotone_at_every_model_size(k):
    """BestN is argmax over K=1..5, so every K must be converged properly."""
    tag, csv = CELLS[0]
    seq, _ = _bound_trace(csv, k, 120)
    rel = np.diff(seq) / np.abs(seq[1:])
    assert rel.min() > -NOISE_FLOOR, (
        f"{tag} K={k}: convergence bound decreased by {rel.min():.3e}")


def test_run_vbhmm_stops_where_the_monotone_rule_says():
    """Guard against the reported F being wired back into the stopping rule.

    The reported F and the convergence bound differ, so if someone re-points
    the check at F this stops matching.
    """
    tag, csv = CELLS[0]
    p = _params()
    d = preprocess_trajectories(str(csv), p)
    _seq, fired = _bound_trace(csv, 3, p.max_iter)
    res = run_vbhmm(d, 3, p)
    assert fired is not None, f"{tag}: rule never fired within max_iter"
    assert res.converged
    assert res.n_iter == fired, (
        f"{tag}: run_vbhmm stopped at iteration {res.n_iter} but the monotone "
        f"rule fires at {fired}")


def test_reported_bound_is_not_the_convergence_bound():
    """The two are deliberately different quantities; if they ever become the
    same, model selection has silently changed (the reported F reproduces
    AAS's BestN 8/8, the convergence bound is the monotone one)."""
    tag, csv = CELLS[0]
    p = _params()
    d = preprocess_trajectories(str(csv), p)
    seq, _ = _bound_trace(csv, 3, 40)
    res = run_vbhmm(d, 3, p)
    assert not np.isclose(res.lower_bound, seq[-1], rtol=1e-9), (
        f"{tag}: reported bound and convergence bound coincide")


def test_rust_and_python_tolerances_agree():
    """The two engines must stop at the same iteration; a drifted constant
    would show up as a small, hard-to-attribute D difference."""
    rs = REPO / "smda-scan" / "smda-scan" / "src" / "vbhmm.rs"
    if not rs.exists():
        pytest.skip("Rust source not present")
    m = re.search(r"const VBEM_CONV_TOL:\s*f64\s*=\s*([0-9eE.+-]+)\s*;",
                  rs.read_text(encoding="utf-8"))
    assert m, "VBEM_CONV_TOL not found in vbhmm.rs"
    assert float(m.group(1)) == VBEM_CONV_TOL
