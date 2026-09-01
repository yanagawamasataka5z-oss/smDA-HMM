"""Variational Bayesian Hidden Markov Model (VBHMM) for diffusion state classification.

Ported from smDA-Python (smda/core/vbhmm.py) with the numba kernel removed:
the Rust engine in smda_scan is the production path here, so numba and
llvmlite are not dependencies of this package.  Everything else is unchanged.


Faithful reimplementation of AAS v2.1 "SimplevbSPT" method.
Classifies 2D diffusion trajectories into discrete diffusion states
using variational Bayes inference on squared displacements (dx²).

References:
  Persson et al., Nature Methods 10, 265–269 (2013) — vbSPT v1.0
  Yanagawa et al., Science Signaling (2018) — the diffusion-state
    application implemented here. doi:10.1126/scisignal.aao1917
  Hiroshima et al., J Mol Biol (2018) — same variational scheme, but a
    different emission model for intensities; NOT what this module does.
    doi:10.1016/j.jmb.2018.02.018
AAS implementation uses direct Dirichlet parameterization (not v1.1 a/B decomposition).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smda_hmm.io import aas_format
from scipy.special import digamma, gammaln

# ---------------------------------------------------------------------------
# Numba availability (same pattern as smda/smt/aas_detector.py)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Spatial dimension of the diffusion model.
#
# This is NOT a parameter.  preprocess_trajectories collapses the data to
# dx2 = dx**2 + dy**2 before any inference runs, so the model is 2D by
# construction.  A `dim` knob would let a caller request 3D and receive a
# silently wrong answer — worse than a dead parameter.  AAS agrees: neither its
# hmm.csv metadata nor settings.csv carries a dimension field (settings.csv is
# headed "Tracking2DSettings").
SPATIAL_DIM = 2

# Relative-change threshold on the convergence bound (see run_vbhmm).
#
# Raised from 1e-8 on 2026-08-31, together with the fix that made the monitored
# quantity monotone.  The old 1e-8 watched a NON-monotone quantity, so it
# tripped at that quantity's turning point rather than at convergence; on
# 260830SampleData/t000028/t0002 it stopped with D 2864 ppm short of the fixed
# point.  Applying 1e-8 to the corrected bound is worse still, because that
# bound decays smoothly instead of turning over: 593-1034 ppm short on all
# eight cells.
#
# The value is bracketed by two measured constraints, both internal to this
# implementation.  Nothing about AAS enters the choice.
#
#   Lower bound on tightness — the stopping point must not be a first-order
#   contributor to the reported D.  Residual distance to the fixed point at the
#   selected model (scripts/b1_conv_threshold_probe.py):
#       1e-8  -> 593-1034 ppm      1e-12 -> 5.7-11.1 ppm
#       1e-10 ->  58- 118 ppm      1e-14 -> 0.7- 1.1 ppm
#
#   Upper bound on tightness — the stopping ITERATION must be reproducible.
#   Below 1e-12 the decision is settled by ULP-level differences in the bound,
#   so the two engines stop at different iterations despite computing the same
#   numbers (scripts/b1_rust_python_parity.py, 16 cells):
#       1e-12 -> 16/16 identical iteration counts, D agrees to 2.4e-08 ppm
#       1e-13 ->  1 cell off by one iteration, D differs by 0.58 ppm
#       1e-14 ->  2 cells off by one iteration, D differs by 0.76 ppm
#   A threshold that makes the answer depend on which engine ran is not a
#   convergence criterion, it is a coin flip near the noise floor.
#
# 1e-12 is the tightest value that keeps the stopping iteration reproducible,
# and it fires in 53-67 iterations, leaving 30+ of headroom against
# max_iter=100 at the selected model.
#
# Two earlier justifications were wrong and are recorded here so they are not
# reintroduced: (a) 1e-12 does NOT put the truncation below the six
# significant digits AAS prints (~1.4-2.7 ppm) -- 5.7-11.1 ppm is above that;
# (b) 1e-14 does NOT exceed max_iter at the selected model (62-80 iterations).
# The "up to 370 iterations" figure was measured over all of K=1..5 and belongs
# to the K=4/K=5 models, which exhaust max_iter at every threshold including
# the old 1e-8.
VBEM_CONV_TOL = 1e-12


@dataclass
class VBHMMParams:
    """Hyperparameters for VBHMM (AAS SimplevbSPT defaults)."""

    n_tilde: float = 1.0        # Gamma prior shape for γ
    c_tilde: float = 0.001      # Gamma prior scale for γ
    w_pi_tilde: float = 1.0     # Dirichlet pseudo-count for initial state
    w_b_tilde: float = 0.01     # Dirichlet pseudo-count for transitions (off-diag)
    mag: float = 30.0           # Diagonal/off-diagonal ratio for initial wB
    max_hidden: int = 5         # Maximum number of states
    min_hidden: int = 1         # Minimum number of states
    max_iter: int = 100         # Maximum VBEM iterations
    num_run: int = 5            # Number of random restarts per model
    timestep: float = -1.0      # Frame interval (s); sentinel -1.0 = unset
    distance_per_pixel: float = -1.0   # μm/pixel; sentinel -1.0 = unset
    frame_minimum: int = 1      # Minimum trajectory length (frames)
    estimate_mode: str = "MaxProb"  # State assignment mode
    is_add_each_trajectory: bool = True   # Scale priors by trajectory count
    is_calc_kl_each: bool = False         # KL calculation mode
    w_pi_scale_mode: int = 1    # 0: unscaled, 1: scaled by n_traj
    w_b_scale_mode: int = 1     # 0: unscaled, 1: scaled (off-diag + diag by mag, AAS P5)

    def validate(self) -> None:
        """Check that measurement-dependent fields are set (not sentinel)."""
        missing = []
        if self.timestep < 0:
            missing.append("timestep")
        if self.distance_per_pixel < 0:
            missing.append("distance_per_pixel")
        if missing:
            raise ValueError(
                f"VBHMMParams fields not set (still at sentinel): {missing}. "
                f"Set values via GUI or ensure settings are loaded."
            )


@dataclass
class VBHMMPriors:
    """Prior parameters for a given N-state model."""

    n: np.ndarray       # (N,) Gamma shape prior
    c: np.ndarray       # (N,) Gamma scale prior
    w_pi: np.ndarray    # (N,) Dirichlet prior for initial state
    w_b: np.ndarray     # (N, N) Dirichlet prior for transitions (full matrix)


@dataclass
class VBHMMState:
    """Variational parameters for a single model."""

    n: np.ndarray       # (N,) Gamma shape
    c: np.ndarray       # (N,) Gamma scale
    w_pi: np.ndarray    # (N,) Dirichlet params for initial state
    w_b: np.ndarray     # (N, N) Dirichlet params for transitions


@dataclass
class VBHMMModelResult:
    """Result for a single N-state model."""

    n_states: int
    lower_bound: float
    ln_zs: float           # lnZQ + lnZq + lnZz
    kl: float              # Total KL
    kl_pi: float
    kl_diffusion: float
    kl_b: float
    state: VBHMMState      # Final variational parameters
    priors: VBHMMPriors    # Priors used
    pst: np.ndarray        # (T, N) posterior state probabilities
    converged: bool
    n_iter: int
    D: np.ndarray          # (N,) Diffusion coefficients [μm²/s]


@dataclass
class VBHMMResult:
    """Full VBHMM analysis result."""

    models: list[VBHMMModelResult]
    best_model: int         # Best N (1-indexed)
    params: VBHMMParams
    state_assignments: dict[int, np.ndarray]  # N -> state array (per frame)


@dataclass
class TrajectoryData:
    """Preprocessed trajectory data for VBHMM."""

    dx2: np.ndarray         # (T,) squared displacements [px²]
    trj_ends: np.ndarray    # (M,) index of last step in each trajectory
    trj_starts: np.ndarray  # (M,) index of first step in each trajectory
    n_trajectories: int     # M
    n_steps: int            # T = total steps
    roi: np.ndarray         # (n_frames,) trajectory IDs per frame
    x_px: np.ndarray        # (n_frames,) x coordinates [px]
    y_px: np.ndarray        # (n_frames,) y coordinates [px]
    frame: np.ndarray       # (n_frames,) frame numbers
    valid_rois: np.ndarray | None = None  # (M,) ROI IDs that passed frame_minimum filter


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

def preprocess_trajectories(
    csv_path: str | Path,
    params: VBHMMParams | None = None,
) -> TrajectoryData:
    """Load AAS4 CSV and compute dx² + trajectory boundaries.

    Parameters
    ----------
    csv_path : path to AAS4 data CSV
    params : VBHMM parameters (for frame_minimum filter)

    Returns
    -------
    TrajectoryData with dx², trajectory boundaries, and raw coordinates.
    """
    if params is None:
        raise ValueError(
            "VBHMMParams must be provided by the caller. "
            "Automatic default construction is not allowed."
        )

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Only the first four columns are read: trajectory id, frame, x and y.
    # They are numeric and populated in both AAS formats -- the state columns
    # further right are the ones that carry an empty terminal marker, and this
    # function never looks at them.
    #
    # A gap in any of these four would pass through astype() as NaN, dx2 would
    # come out NaN, and the run would produce a result that looks like a
    # result.  Check instead of finding out later.
    _needed = ["No", "Time [frame]", "xg [px]", "yg [px]"]
    if df.shape[1] < 4:
        raise ValueError(
            f"{csv_path}: {df.shape[1]} columns; the first four "
            f"({', '.join(_needed)}) are required."
        )
    _block = df.iloc[:, :4]
    _missing = _block.isna()
    if _missing.to_numpy().any():
        _rows, _cols = np.nonzero(_missing.to_numpy())
        _first = [
            f"line {int(r) + 2}, column {int(c)} "
            f"({str(df.columns[int(c)])!r})"
            for r, c in list(zip(_rows, _cols))[:5]
        ]
        raise ValueError(
            f"{csv_path}: {int(_missing.to_numpy().sum())} empty or "
            f"non-numeric cell(s) among the first four columns "
            f"({', '.join(_needed)}), which hold the trajectory id, frame "
            f"number and position.\n  " + "\n  ".join(_first)
            + "\n  These are not filled in: a gap here becomes NaN, and "
              "every squared displacement computed from it would be NaN too."
        )

    roi = df.iloc[:, 0].values.astype(np.float64)   # No (molecule ID)
    frame = df.iloc[:, 1].values.astype(np.float64)  # Time [frame]
    x_px = df.iloc[:, 2].values.astype(np.float64)   # xg [px]
    y_px = df.iloc[:, 3].values.astype(np.float64)   # yg [px]

    return _preprocess_from_arrays(roi, frame, x_px, y_px, params)


def _preprocess_from_arrays(
    roi: np.ndarray,
    frame: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    params: VBHMMParams,
) -> TrajectoryData:
    """Core preprocessing from arrays.

    AAS computes dx² in μm² (converts pixel coordinates using distancePerPixel).
    """
    unique_rois = np.unique(roi[~np.isnan(roi)])
    scale = params.distance_per_pixel  # μm/px

    all_dx2 = []
    trj_starts = []
    trj_ends = []
    valid_roi_ids = []
    step_offset = 0

    for r in unique_rois:
        mask = roi == r
        idx = np.where(mask)[0]
        if len(idx) < params.frame_minimum + 1:
            continue

        x_trj = x_px[idx]
        y_trj = y_px[idx]

        # dx² in μm²: convert px → μm then square
        dx = np.diff(x_trj) * scale
        dy = np.diff(y_trj) * scale
        dx2_trj = dx**2 + dy**2

        n_steps = len(dx2_trj)
        if n_steps == 0:
            continue

        all_dx2.append(dx2_trj)
        trj_starts.append(step_offset)
        trj_ends.append(step_offset + n_steps - 1)
        valid_roi_ids.append(r)
        step_offset += n_steps

    dx2 = np.concatenate(all_dx2)
    trj_starts_arr = np.array(trj_starts, dtype=np.int64)
    trj_ends_arr = np.array(trj_ends, dtype=np.int64)
    valid_rois_arr = np.array(valid_roi_ids, dtype=np.float64)

    return TrajectoryData(
        dx2=dx2,
        trj_ends=trj_ends_arr,
        trj_starts=trj_starts_arr,
        n_trajectories=len(trj_starts),
        n_steps=len(dx2),
        roi=roi,
        x_px=x_px,
        y_px=y_px,
        frame=frame,
        valid_rois=valid_rois_arr,
    )


# ---------------------------------------------------------------------------
# 2. Prior construction
# ---------------------------------------------------------------------------

def _validate_scale_modes(params: VBHMMParams) -> None:
    """Reject unimplemented scale modes instead of silently ignoring them.

    Only 0 and 1 exist.  Any other value used to fall through to the ``else``
    branch and produce unscaled priors with a zero diagonal — silently.  S69
    recorded the cost of that: a mis-set mode flipped BestN and gave a 23% D
    error (docs/hadoff/session69_handoff.md section 4).
    """
    if params.w_pi_scale_mode not in (0, 1):
        raise ValueError(
            f"w_pi_scale_mode={params.w_pi_scale_mode} is not implemented "
            f"(valid: 0 = unscaled, 1 = scaled by trajectory count)."
        )
    if params.w_b_scale_mode not in (0, 1):
        raise ValueError(
            f"w_b_scale_mode={params.w_b_scale_mode} is not implemented "
            f"(valid: 0 = unscaled, 1 = scaled, AAS P5). "
            f"Note: S69 renumbered these — the old mode 2 is the current mode 1."
        )


def _build_priors(n_states: int, n_trajectories: int, params: VBHMMParams) -> VBHMMPriors:
    """Build prior parameters for N-state model (scaled by M for M-step).

    n, c: always scaled by M. w_pi, w_b: controlled by scale_mode params.
    """
    _validate_scale_modes(params)

    M = n_trajectories if params.is_add_each_trajectory else 1

    prior_n = np.full(n_states, M * params.n_tilde)
    prior_c = np.full(n_states, M * params.c_tilde)

    # w_pi: mode 0 = unscaled, mode 1 = scaled by M
    s_pi = M if params.w_pi_scale_mode == 1 else 1
    prior_w_pi = np.full(n_states, s_pi * params.w_pi_tilde)

    # w_b: mode 0 = unscaled, mode 1 = scaled (AAS P5)
    s_b = M if params.w_b_scale_mode == 1 else 1
    prior_w_b = np.full((n_states, n_states), s_b * params.w_b_tilde)
    if params.w_b_scale_mode == 1:
        # diagonal = mag × wBTilde × M (AAS P5)
        np.fill_diagonal(prior_w_b, params.mag * params.w_b_tilde * M)
    else:
        np.fill_diagonal(prior_w_b, 0.0)

    return VBHMMPriors(n=prior_n, c=prior_c, w_pi=prior_w_pi, w_b=prior_w_b)


def _build_base_priors(n_states: int, params: VBHMMParams) -> VBHMMPriors:
    """Build unscaled base priors for KL computation (isCalcKlEach=false)."""
    prior_n = np.full(n_states, params.n_tilde)
    prior_c = np.full(n_states, params.c_tilde)
    prior_w_pi = np.full(n_states, params.w_pi_tilde)

    prior_w_b = np.full((n_states, n_states), params.w_b_tilde)
    # diagonal = 0 for base priors (diagonal补完 is done in _compute_lower_bound)
    np.fill_diagonal(prior_w_b, 0.0)

    return VBHMMPriors(n=prior_n, c=prior_c, w_pi=prior_w_pi, w_b=prior_w_b)


# ---------------------------------------------------------------------------
# 3. K-means++ initialization
# ---------------------------------------------------------------------------

def kmeans_init(
    data: TrajectoryData,
    n_states: int,
    priors: VBHMMPriors,
    params: VBHMMParams,
) -> VBHMMState:
    """Initialize VBHMM parameters using deterministic K-means++ on dx².

    Identical to smda-scan/smda-scan/src/vbhmm.rs:kmeans_init.  See
    _deterministic_kmeans_1d for why this is not the stochastic k-means++ the
    Python side used until 2026-08-31.

    Parameters
    ----------
    data : preprocessed trajectory data
    n_states : number of HMM states
    priors : prior parameters
    params : VBHMM hyperparameters

    Returns
    -------
    VBHMMState with initialized variational parameters.
    """
    dx2 = data.dx2
    dim = SPATIAL_DIM
    dt = params.timestep

    if n_states == 1:
        # Single state: all data in one cluster
        n_init = priors.n.copy()
        n_init[0] += (dim / 2) * len(dx2)
        c_init = priors.c.copy()
        c_init[0] += np.sum(dx2)
        w_pi_init = priors.w_pi.copy()
        w_pi_init[0] += data.n_trajectories
        w_b_init = np.array([[data.n_steps - data.n_trajectories]])
        return VBHMMState(n=n_init, c=c_init, w_pi=w_pi_init, w_b=w_b_init)

    labels = _deterministic_kmeans_1d(dx2, n_states)

    # Compute per-cluster statistics
    n_init = priors.n.copy()
    c_init = priors.c.copy()
    w_pi_init = priors.w_pi.copy()

    cluster_means = np.zeros(n_states)
    for k in range(n_states):
        mask_k = labels == k
        count_k = np.sum(mask_k)
        # Sequential accumulation in index order, matching Rust's
        #   let sum: f64 = (0..t).filter(|i| labels[i]==k).map(|i| dx2[i]).sum()
        # np.sum uses pairwise summation and lands 1-2 ULP away, which is
        # enough to make the two implementations diverge over ~30 iterations.
        sum_dx2_k = 0.0
        for _v in dx2[mask_k]:
            sum_dx2_k += float(_v)
        cluster_means[k] = sum_dx2_k / max(count_k, 1)

        n_init[k] += (dim / 2) * count_k
        c_init[k] += sum_dx2_k

    # Sort clusters by mean dx² (ascending D)
    sort_idx = np.argsort(cluster_means)
    n_init = n_init[sort_idx]
    c_init = c_init[sort_idx]
    # Re-label
    new_labels = np.empty_like(labels)
    for new_k, old_k in enumerate(sort_idx):
        new_labels[labels == old_k] = new_k
    labels = new_labels

    # Initial wPi from trajectory start assignments
    for m in range(data.n_trajectories):
        start_idx = data.trj_starts[m]
        k = labels[start_idx]
        w_pi_init[k] += 1.0

    # Initial wB: diagonal-dominant transition matrix
    w_b_init = priors.w_b.copy()
    # Count transitions within each trajectory
    trans_counts = np.zeros((n_states, n_states))
    for m in range(data.n_trajectories):
        s = data.trj_starts[m]
        e = data.trj_ends[m]
        for t in range(s, e):
            i_state = labels[t]
            j_state = labels[t + 1]
            trans_counts[i_state, j_state] += 1.0

    # Apply mag to diagonal
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                w_b_init[i, j] += params.mag * max(trans_counts[i, j], 1.0)
            else:
                w_b_init[i, j] += max(trans_counts[i, j], 1.0)

    return VBHMMState(n=n_init, c=c_init, w_pi=w_pi_init, w_b=w_b_init)


def _deterministic_kmeans_1d(dx2: np.ndarray, k: int) -> np.ndarray:
    """Deterministic K-means++ on dx², byte-for-byte equivalent to Rust.

    Port of smda-scan/smda-scan/src/vbhmm.rs:kmeans_init (S41 Phase 5.A,
    commit 2653179).  Read the history before changing either copy:

    * Both sides were originally stochastic (LCG in Rust, rng.integers /
      rng.choice in Python).  S41 made **Rust** deterministic to guarantee
      reproducibility; the Python copy was simply never updated and stayed
      stochastic until 2026-08-31.  The divergence had been on the books since
      S10 ("VBHMM 状態割当の Python-Rust 差異: K-means++ 乱数初期化が異なる").
      It was an update that was missed, not a deliberate second opinion —
      **the Python path is not an independent reference implementation.**
    * median + farthest-point is NOT known to be what AAS does.  S41 chose it
      to obtain determinism; the audit in session43_smt_seed_audit.md records
      AAS's initialisation as "不明" (unknown), and S41 itself measured that
      switching to it left the LB difference against AAS essentially unchanged.

    Steps (each matching the Rust source line for line):
      1. centre 1 = the sample closest to the median of log(dx2)
      2. centres 2..k = farthest-point heuristic on log(dx2)
      3. convert the centres to linear space with exp()
      4. Lloyd, at most 50 rounds, **on linear dx2**, stopping when every
         centre moves by <= 1e-10
      5. one final assignment pass, then relabel by ascending cluster mean

    Note steps 1-2 work in log space while step 4 works in linear space.  That
    asymmetry is what Rust does, so it is what this function does.
    """
    t = len(dx2)
    log_dx2 = np.log(np.maximum(dx2, 1e-300))

    # 1. first centre: the point closest to the median of log(dx2)
    sorted_log = np.sort(log_dx2)
    median_val = sorted_log[t // 2]
    centres = [float(log_dx2[int(np.argmin(np.abs(log_dx2 - median_val)))])]

    # 2. remaining centres: farthest point by min squared distance
    for _ in range(1, k):
        d2 = np.min(
            np.stack([(log_dx2 - c) ** 2 for c in centres], axis=1), axis=1)
        centres.append(float(log_dx2[int(np.argmax(d2))]))

    # 3. log -> linear
    lin = np.exp(np.asarray(centres, dtype=np.float64))

    # 4. Lloyd on linear dx2
    labels = np.zeros(t, dtype=np.int64)
    for _ in range(50):
        labels = np.argmin(np.abs(dx2[:, None] - lin[None, :]), axis=1)
        sums = np.zeros(k)
        counts = np.zeros(k, dtype=np.int64)
        for c_idx in range(k):
            mask = labels == c_idx
            counts[c_idx] = int(mask.sum())
            sums[c_idx] = float(dx2[mask].sum())
        new_lin = np.where(counts > 0, sums / np.maximum(counts, 1), lin)
        changed = bool(np.any(np.abs(new_lin - lin) > 1e-10))
        lin = new_lin
        if not changed:
            break

    # 5. final assignment, then relabel by ascending cluster mean of dx2
    labels = np.argmin(np.abs(dx2[:, None] - lin[None, :]), axis=1)
    means = np.array([dx2[labels == c].mean() if np.any(labels == c) else 0.0
                      for c in range(k)])
    order = np.argsort(means)
    remap = np.empty(k, dtype=np.int64)
    for new_k, old_k in enumerate(order):
        remap[old_k] = new_k
    return remap[labels]


# ---------------------------------------------------------------------------
# 4. Forward-Backward
# ---------------------------------------------------------------------------

# -- Numba JIT implementation --


def _forward_backward_numpy(Q, H, trj_starts, trj_ends):
    """All-trajectory Forward-Backward (pure NumPy)."""
    N = Q.shape[0]
    T = H.shape[0]
    M = len(trj_starts)

    pst = np.zeros((T, N), dtype=np.float64)
    wA = np.zeros((N, N), dtype=np.float64)
    lnZz = 0.0

    for m in range(M):
        s = trj_starts[m]
        e = trj_ends[m]
        Tm = e - s + 1

        # Forward
        alpha = np.zeros((Tm, N), dtype=np.float64)
        scale = np.zeros(Tm, dtype=np.float64)
        alpha[0] = H[s]
        scale[0] = np.sum(alpha[0])
        if scale[0] > 0:
            alpha[0] /= scale[0]
        else:
            alpha[0] = 1.0 / N
            scale[0] = 1e-300

        for t in range(1, Tm):
            alpha[t] = (alpha[t - 1] @ Q) * H[s + t]
            scale[t] = np.sum(alpha[t])
            if scale[t] > 0:
                alpha[t] /= scale[t]
            else:
                alpha[t] = 1.0 / N
                scale[t] = 1e-300

        # Backward
        beta = np.zeros((Tm, N), dtype=np.float64)
        beta[Tm - 1] = 1.0
        for t in range(Tm - 2, -1, -1):
            beta[t] = Q @ (H[s + t + 1] * beta[t + 1])
            s_beta = np.sum(beta[t])
            if s_beta > 0:
                beta[t] /= s_beta

        # Posterior
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma /= gamma_sum
        pst[s:e + 1] = gamma

        # Transition counts
        for t in range(Tm - 1):
            xi = (alpha[t][:, None] * Q) * (H[s + t + 1] * beta[t + 1])[None, :]
            xi_sum = np.sum(xi)
            if xi_sum > 0:
                xi /= xi_sum
            wA += xi

        lnZz += np.sum(np.log(np.maximum(scale, 1e-300)))

    return lnZz, wA, pst


def forward_backward(
    ln_h: np.ndarray,
    ln_q: np.ndarray,
    trj_starts: np.ndarray,
    trj_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standard Forward-Backward algorithm with trajectory boundaries.

    Dispatches to Numba JIT or NumPy fallback based on availability.

    Parameters
    ----------
    ln_h : (T, N) log emission probabilities (includes initial state prob at trj starts)
    ln_q : (N, N) log transition probabilities (digamma-based)
    trj_starts : (M,) start indices
    trj_ends : (M,) end indices

    Returns
    -------
    pst : (T, N) posterior state probabilities γ_t(j)
    wA : (N, N) expected transition counts Σ_t ξ_t(i,j)
    lnZz : float, sum of log normalization constants
    """
    # Normalize Q and H before passing to FB kernel
    ln_q_max = np.max(ln_q)
    Q = np.exp(ln_q - ln_q_max)

    ln_h_max = np.max(ln_h, axis=1, keepdims=True)  # (T, 1)
    H = np.exp(ln_h - ln_h_max)  # (T, N)

    # NumPy only.  smDA-Python also carried a numba kernel here; the Rust
    # engine is the production path in this package, so the numba dependency
    # (numba + llvmlite, ~113 MB) buys nothing and is not shipped.
    lnZz, wA, pst = _forward_backward_numpy(Q, H, trj_starts, trj_ends)

    return pst, wA, lnZz


# ---------------------------------------------------------------------------
# 5. E-step
# ---------------------------------------------------------------------------

def _e_step(
    dx2: np.ndarray,
    state: VBHMMState,
    trj_starts: np.ndarray,
    trj_ends: np.ndarray,
    dim: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """E-step: compute emission/transition probs and run Forward-Backward.

    Returns
    -------
    pst : (T, N) posterior
    wA : (N, N) transition counts
    lnZz : log normalization
    lnZQ : log transition normalization term
    lnZq : log emission normalization term
    """
    T = len(dx2)
    N = len(state.n)
    M = len(trj_starts)

    # --- Emission probabilities (log) ---
    # lnH0[j] = (dim/2) * (digamma(n[j]) - log(pi * c[j]))
    ln_h0 = (dim / 2) * (digamma(state.n) - np.log(np.pi * state.c))  # (N,)
    # lnH[t, j] = lnH0[j] - (n[j] / c[j]) * dx2[t]
    gamma_rate = state.n / state.c  # (N,)
    ln_h = ln_h0[None, :] - gamma_rate[None, :] * dx2[:, None]  # (T, N)

    # Add initial state probability at trajectory starts
    w_pi_sum = np.sum(state.w_pi)
    ln_pi = digamma(state.w_pi) - digamma(w_pi_sum)  # (N,)
    for m in range(M):
        ln_h[trj_starts[m]] += ln_pi

    # --- Transition probabilities (log) ---
    # lnQ[i,j] = digamma(wB[i,j]) - digamma(sum(wB[i,:]))
    w_b_rowsum = np.sum(state.w_b, axis=1)  # (N,)
    ln_q = digamma(state.w_b) - digamma(w_b_rowsum)[:, None]  # (N, N)

    # --- lnZQ ---
    ln_q_max = np.max(ln_q)
    total_transitions = T - M  # total steps minus trajectory starts
    lnZQ = total_transitions * ln_q_max

    # --- lnZq ---
    ln_h_max = np.max(ln_h, axis=1)  # (T,)
    lnZq = np.sum(ln_h_max)

    # --- Forward-Backward ---
    pst, wA, lnZz = forward_backward(ln_h, ln_q, trj_starts, trj_ends)

    return pst, wA, lnZz, lnZQ, lnZq


# ---------------------------------------------------------------------------
# 6. M-step
# ---------------------------------------------------------------------------

def _m_step(
    dx2: np.ndarray,
    pst: np.ndarray,
    wA: np.ndarray,
    priors: VBHMMPriors,
    trj_starts: np.ndarray,
    dim: int = 2,
) -> VBHMMState:
    """M-step: update variational parameters.

    Parameters
    ----------
    dx2 : (T,) squared displacements
    pst : (T, N) posterior state probabilities
    wA : (N, N) expected transition counts
    priors : prior parameters
    trj_starts : trajectory start indices
    dim : spatial dimension

    Returns
    -------
    Updated VBHMMState.
    """
    N = pst.shape[1]

    # Initial state counts
    pst_starts = pst[trj_starts]  # (M, N)
    w_pi = priors.w_pi + np.sum(pst_starts, axis=0)  # (N,)

    # Transition counts
    w_b = priors.w_b + wA  # (N, N)

    # Emission parameters
    pst_sum = np.sum(pst, axis=0)           # (N,)
    pst_dx2_sum = pst.T @ dx2               # (N,)
    n = priors.n + (dim / 2) * pst_sum      # (N,)
    c = priors.c + pst_dx2_sum              # (N,)

    return VBHMMState(n=n, c=c, w_pi=w_pi, w_b=w_b)


# ---------------------------------------------------------------------------
# 7. Lower Bound (KL terms)
# ---------------------------------------------------------------------------

def _kl_dirichlet(w: np.ndarray, u: np.ndarray) -> float:
    """KL divergence between two Dirichlet distributions.

    KL[Dir(w) || Dir(u)] for a single Dirichlet (1D parameter vector).
    """
    # Clip parameters to avoid gammaln/digamma with near-zero or negative values
    _EPS = 1e-300
    w = np.maximum(w, _EPS)
    u = np.maximum(u, _EPS)
    w0 = np.sum(w)
    u0 = np.sum(u)
    kl = gammaln(w0) - gammaln(u0)
    kl += np.sum(gammaln(u) - gammaln(w))
    kl += np.sum((w - u) * (digamma(w) - digamma(w0)))
    if not np.isfinite(kl):
        import warnings
        warnings.warn(f"KL divergence non-finite ({kl}), model rejected")
        return np.inf
    return kl


def _kl_pi(w_pi: np.ndarray, prior_w_pi: np.ndarray) -> float:
    """KL divergence for initial state distribution."""
    return _kl_dirichlet(w_pi, prior_w_pi)


def _kl_diffusion(n: np.ndarray, c: np.ndarray,
                   prior_n: np.ndarray, prior_c: np.ndarray) -> float:
    """KL divergence for emission (Gamma) parameters."""
    _EPS = 1e-300
    kl = 0.0
    for j in range(len(n)):
        n_j = max(n[j], _EPS)
        c_j = max(c[j], _EPS)
        prior_c_j = max(prior_c[j], _EPS)
        prior_n_j = max(prior_n[j], _EPS)
        kl_j = (prior_n_j * np.log(c_j / prior_c_j)
                - n_j * (1.0 - prior_c_j / c_j)
                - gammaln(n_j) + gammaln(prior_n_j)
                + (n_j - prior_n_j) * digamma(n_j))
        if np.isfinite(kl_j):
            kl += kl_j
        else:
            import warnings
            warnings.warn(f"KL emission term[{j}] non-finite ({kl_j}), "
                          "model rejected")
            return np.inf
    return kl


def _kl_transition(w_b: np.ndarray, prior_w_b_full: np.ndarray) -> float:
    """KL divergence for transition matrix (direct Dirichlet, per row)."""
    N = w_b.shape[0]
    kl = 0.0
    for i in range(N):
        kl += _kl_dirichlet(w_b[i], prior_w_b_full[i])
    return kl


def _compute_lower_bound(
    lnZQ: float,
    lnZq: float,
    lnZz: float,
    state: VBHMMState,
    base_priors: VBHMMPriors,
    mag: float = 30.0,
) -> tuple[float, float, float, float, float]:
    """Compute variational lower bound F.

    Uses base (unscaled) priors for KL computation when isCalcKlEach=false.
    AAS scales priors by M for the M-step but uses original hyperparameters for KL.

    Parameters
    ----------
    base_priors : unscaled priors (nTilde, cTilde, etc.) for KL computation
    mag : diagonal/off-diagonal ratio for w_b prior (used in KL computation)

    Returns
    -------
    F, kl_pi, kl_diff, kl_b, ln_zs
    """
    N = len(state.n)

    kl_pi = _kl_pi(state.w_pi, base_priors.w_pi)
    kl_diff = _kl_diffusion(state.n, state.c, base_priors.n, base_priors.c)

    if N == 1:
        kl_b = 0.0
    else:
        # Full prior wB: off-diagonal from base_priors, diagonal = off_diag * mag
        # (AAS uses mag-scaled diagonal for KL prior, verified in Session 41 Phase 4)
        prior_w_b_full = base_priors.w_b.copy()
        off_diag = base_priors.w_b[0, 1]
        np.fill_diagonal(prior_w_b_full, off_diag * mag)
        kl_b = _kl_transition(state.w_b, prior_w_b_full)

    ln_zs = lnZQ + lnZq + lnZz
    kl_total = kl_pi + kl_diff + kl_b
    F = ln_zs - kl_total

    return F, kl_pi, kl_diff, kl_b, ln_zs


def _compute_per_trajectory_kl(
    dx2: np.ndarray,
    state: VBHMMState,
    trj_starts: np.ndarray,
    trj_ends: np.ndarray,
    base_priors: VBHMMPriors,
    dim: int = 2,
    mag: float = 30.0,
) -> tuple[float, float, float, float]:
    """Compute KL per trajectory and sum (isCalcKlEach=True).

    For each trajectory m, builds a per-trajectory posterior from
    base_priors + that trajectory's sufficient statistics, then computes
    KL(posterior_m || base_prior) and sums across all trajectories.

    Returns kl_total, kl_pi_total, kl_diff_total, kl_b_total.
    """
    N = len(state.n)
    M = len(trj_starts)

    # Reconstruct emission/transition from converged state
    ln_h0 = (dim / 2) * (digamma(state.n) - np.log(np.pi * state.c))
    gamma_rate = state.n / state.c
    ln_h = ln_h0[None, :] - gamma_rate[None, :] * dx2[:, None]
    w_pi_sum = np.sum(state.w_pi)
    ln_pi = digamma(state.w_pi) - digamma(w_pi_sum)
    for m in range(M):
        ln_h[trj_starts[m]] += ln_pi

    w_b_rowsum = np.sum(state.w_b, axis=1)
    ln_q = digamma(state.w_b) - digamma(w_b_rowsum)[:, None]
    ln_q_max = np.max(ln_q)
    Q = np.exp(ln_q - ln_q_max)
    ln_h_max = np.max(ln_h, axis=1, keepdims=True)
    H = np.exp(ln_h - ln_h_max)

    # Base prior w_b with diagonal = off_diag * mag
    bp_wb_full = base_priors.w_b.copy()
    if N > 1:
        off_diag = base_priors.w_b[0, 1]
        np.fill_diagonal(bp_wb_full, off_diag * mag)

    kl_pi_total = 0.0
    kl_diff_total = 0.0
    kl_b_total = 0.0

    for m in range(M):
        s = trj_starts[m]
        e = trj_ends[m]
        L = e - s + 1

        # Forward pass
        alpha = np.zeros((L, N))
        alpha[0] = H[s]
        scale = np.zeros(L)
        scale[0] = alpha[0].sum()
        if scale[0] > 1e-300:
            alpha[0] /= scale[0]
        for t in range(1, L):
            alpha[t] = (alpha[t - 1] @ Q) * H[s + t]
            scale[t] = alpha[t].sum()
            if scale[t] > 1e-300:
                alpha[t] /= scale[t]

        # Backward pass
        beta = np.zeros((L, N))
        beta[L - 1] = 1.0
        for t in range(L - 2, -1, -1):
            beta[t] = Q @ (H[s + t + 1] * beta[t + 1])
            if scale[t + 1] > 1e-300:
                beta[t] /= scale[t + 1]

        # Posterior pst_m
        pst_m = alpha * beta
        row_sums = pst_m.sum(axis=1, keepdims=True)
        pst_m = np.where(row_sums > 1e-300, pst_m / row_sums, 1.0 / N)

        # Per-trajectory transition counts xi_m
        wA_m = np.zeros((N, N))
        for t in range(L - 1):
            xi = alpha[t][:, None] * Q * (H[s + t + 1] * beta[t + 1])[None, :]
            xi_sum = xi.sum()
            if xi_sum > 1e-300:
                xi /= xi_sum
            wA_m += xi

        # Per-trajectory posterior parameters
        # Emission (Gamma)
        n_m = base_priors.n + (dim / 2) * pst_m.sum(axis=0)
        c_m = base_priors.c.copy()
        for j in range(N):
            c_m[j] += (pst_m[:, j] * dx2[s:e + 1]).sum()

        # Initial state (Dirichlet)
        w_pi_m = base_priors.w_pi + pst_m[0]

        # Transition (Dirichlet per row)
        w_b_m = bp_wb_full + wA_m

        # KL computation
        kl_pi_total += _kl_pi(w_pi_m, base_priors.w_pi)
        kl_diff_total += _kl_diffusion(n_m, c_m, base_priors.n, base_priors.c)
        if N > 1:
            kl_b_total += _kl_transition(w_b_m, bp_wb_full)

    kl_total = kl_pi_total + kl_diff_total + kl_b_total
    return kl_total, kl_pi_total, kl_diff_total, kl_b_total


# ---------------------------------------------------------------------------
# 8. VBEM iteration
# ---------------------------------------------------------------------------

def vbem_step(
    dx2: np.ndarray,
    state: VBHMMState,
    priors: VBHMMPriors,
    base_priors: VBHMMPriors,
    trj_starts: np.ndarray,
    trj_ends: np.ndarray,
    dim: int = 2,
    mag: float = 30.0,
) -> tuple[VBHMMState, np.ndarray, float, float, float, float, float]:
    """Single VBEM iteration: E-step → M-step → Lower Bound.

    Parameters
    ----------
    priors : scaled priors (M × tilde) for M-step updates
    base_priors : unscaled priors (tilde) for KL computation
    mag : diagonal/off-diagonal ratio for w_b prior

    Returns
    -------
    new_state, pst, F, kl_pi, kl_diff, kl_b, ln_zs
    """
    # E-step
    pst, wA, lnZz, lnZQ, lnZq = _e_step(
        dx2, state, trj_starts, trj_ends, dim)

    # M-step (uses scaled priors)
    new_state = _m_step(dx2, pst, wA, priors, trj_starts, dim)

    # Lower Bound (uses base priors for KL)
    F, kl_pi, kl_diff, kl_b, ln_zs = _compute_lower_bound(
        lnZQ, lnZq, lnZz, new_state, base_priors, mag)

    return new_state, pst, F, kl_pi, kl_diff, kl_b, ln_zs


# ---------------------------------------------------------------------------
# 9. Single model convergence
# ---------------------------------------------------------------------------

def run_vbhmm(
    data: TrajectoryData,
    n_states: int,
    params: VBHMMParams,
) -> VBHMMModelResult:
    """Run VBHMM for a single N-state model until convergence.

    Parameters
    ----------
    data : preprocessed trajectory data
    n_states : number of HMM states
    params : VBHMM hyperparameters

    Returns
    -------
    VBHMMModelResult with converged parameters and lower bound.
    """
    priors = _build_priors(n_states, data.n_trajectories, params)
    base_priors = _build_base_priors(n_states, params)
    # Always use base_priors for KL during VBEM convergence checking.
    # Per-trajectory KL (is_calc_kl_each) is computed post-convergence.
    kl_priors = base_priors
    state = kmeans_init(data, n_states, priors, params)

    F_old = -np.inf
    F_conv_old = -np.inf
    converged = False
    pst = None

    # The reported bound F (below) measures its KL against the UNSCALED priors
    # while the M-step fits with the M-scaled ones, so it is the bound of a
    # different model than the one being fitted and the VBEM monotonicity
    # theorem does not cover it — measured, it rises, peaks and then descends
    # to the fixed point.  Convergence is therefore judged on F_conv, which
    # evaluates lnZ and the KL at the same state and against the same scaled
    # priors the M-step uses.  That combination is monotone to machine epsilon
    # (worst decrease 1.5e-10 on a bound of 5e4, i.e. 2e-15 relative).
    #
    # Both factors matter: correcting only the prior scale, or only the
    # evaluation point, leaves the sequence non-monotone.
    # scripts/b1_monotonicity_matrix.py, tests/test_vbhmm_monotone_bound.py
    #
    # F itself is left exactly as it was: it feeds model selection and hmm.csv,
    # and it is what reproduces AAS's BestN 8/8.
    for iteration in range(1, params.max_iter + 1):
        old_n = state.n.copy()
        old_c = state.c.copy()

        pst, wA, lnZz, lnZQ, lnZq = _e_step(
            data.dx2, state, data.trj_starts, data.trj_ends, SPATIAL_DIM)
        F_conv = _compute_lower_bound(
            lnZQ, lnZq, lnZz, state, priors, params.mag)[0]
        state = _m_step(
            data.dx2, pst, wA, priors, data.trj_starts, SPATIAL_DIM)
        F, kl_pi, kl_diff, kl_b, ln_zs = _compute_lower_bound(
            lnZQ, lnZq, lnZz, state, kl_priors, params.mag)

        # Convergence check
        if F_conv_old != -np.inf:
            rel_change = abs(F_conv - F_conv_old) / max(abs(F_conv), 1e-300)
            param_change = max(
                np.max(np.abs(state.n - old_n) / np.maximum(np.abs(old_n), 1e-300)),
                np.max(np.abs(state.c - old_c) / np.maximum(np.abs(old_c), 1e-300)),
            )
            if rel_change < VBEM_CONV_TOL and param_change < 1e-2:
                converged = True
                F_old = F
                break

        F_conv_old = F_conv
        F_old = F

    # Diffusion coefficients [μm²/s]
    dt = params.timestep
    # E[D] = c / [4·dt·(n-1)] — true posterior expectation under γ ~ Gamma(n, c)
    # (paper Eq. 16, matches AAS2 to floating-point precision)
    D = state.c / (4.0 * dt * (state.n - 1.0))  # μm²/s

    # Per-trajectory KL (isCalcKlEach=True): recompute KL post-convergence
    if params.is_calc_kl_each:
        kl_total, kl_pi, kl_diff, kl_b = _compute_per_trajectory_kl(
            data.dx2, state, data.trj_starts, data.trj_ends,
            base_priors, SPATIAL_DIM, params.mag,
        )
        F_old = ln_zs - kl_total
    else:
        kl_total = kl_pi + kl_diff + kl_b

    return VBHMMModelResult(
        n_states=n_states,
        lower_bound=F_old,
        ln_zs=ln_zs,
        kl=kl_total,
        kl_pi=kl_pi,
        kl_diffusion=kl_diff,
        kl_b=kl_b,
        state=state,
        priors=priors,
        pst=pst,
        converged=converged,
        n_iter=iteration,
        D=D,
    )


# ---------------------------------------------------------------------------
# 10. State assignment
# ---------------------------------------------------------------------------

def assign_states(
    result: VBHMMModelResult,
    data: TrajectoryData,
) -> np.ndarray:
    """Assign states to each frame using MaxProb method.

    Returns
    -------
    states : (n_frames,) array, 1-indexed states sorted by D ascending.
             0 for last frame of each trajectory.
    """
    N = result.n_states
    pst = result.pst  # (T_steps, N)

    # Sort states by D ascending
    d_order = np.argsort(result.D)  # indices that sort D ascending
    # d_order[new_state] = old_state
    # We need: old_to_new[old_state] = new_state + 1 (1-indexed)
    old_to_new = np.zeros(N, dtype=np.int64)
    for new_k, old_k in enumerate(d_order):
        old_to_new[old_k] = new_k + 1

    # MaxProb: argmax over states for each step
    step_states = np.argmax(pst, axis=1)  # (T,) 0-indexed old state
    step_states_sorted = old_to_new[step_states]  # (T,) 1-indexed new state

    # Map steps back to frames — only iterate valid (filtered) ROIs
    # to keep step_idx aligned with step_states_sorted
    if data.valid_rois is not None:
        iter_rois = data.valid_rois
    else:
        # Fallback for legacy TrajectoryData without valid_rois
        iter_rois = np.unique(data.roi[~np.isnan(data.roi)])

    frame_states = np.zeros(len(data.roi), dtype=np.int64)

    step_idx = 0
    for r in iter_rois:
        mask = data.roi == r
        idx = np.where(mask)[0]
        n_frames_trj = len(idx)
        n_steps_trj = n_frames_trj - 1

        if n_steps_trj <= 0:
            continue

        # Assign step states to frames (step t corresponds to frame t)
        for t in range(n_steps_trj):
            frame_states[idx[t]] = step_states_sorted[step_idx + t]

        # Last frame of trajectory: state 0
        frame_states[idx[-1]] = 0

        step_idx += n_steps_trj

    return frame_states


# ---------------------------------------------------------------------------
# 11. Full analysis pipeline
# ---------------------------------------------------------------------------

def run_vbhmm_analysis(
    csv_path: str | Path,
    params: VBHMMParams | None = None,
    seed: int | None = 42,
    progress_callback: callable | None = None,
    num_run: int | None = None,
) -> VBHMMResult:
    """Run full VBHMM analysis: all N-state models + model selection.

    .. deprecated::
        The production pipeline uses the Rust backend (_run_vbhmm_rust).  This
        function is retained for reference and testing only.

    As of 2026-08-31 the initialisation is deterministic and identical to Rust
    (see _deterministic_kmeans_1d).  ``seed`` and ``num_run`` therefore have no
    effect on the result; they are accepted only so existing callers keep
    working.  Before that date this path used stochastic K-means++, which is
    why it disagreed with Rust — an update missed in S41, not a deliberate
    second implementation.

    Parameters
    ----------
    csv_path : path to AAS4 data CSV
    params : VBHMM hyperparameters (defaults to AAS v2.1 settings)
    seed : ignored (kept for call compatibility; initialisation is deterministic)
    progress_callback : callable(current_n, max_n, lower_bound) for progress updates
    num_run : ignored (kept for call compatibility; restarts are pointless when
        the initialisation is deterministic)

    Returns
    -------
    VBHMMResult with all model results and state assignments.
    """
    import warnings
    warnings.warn(
        "run_vbhmm_analysis (Python) is deprecated. "
        "Use the Rust backend via _run_vbhmm_rust. "
        "Initialisation is now deterministic and identical in both.",
        DeprecationWarning,
        stacklevel=2,
    )
    if params is None:
        raise ValueError(
            "VBHMMParams must be provided by the caller. "
            "Automatic default construction is not allowed."
        )
    params.validate()

    data = preprocess_trajectories(csv_path, params)

    models = []
    for n_states in range(params.min_hidden, params.max_hidden + 1):
        # Initialisation is deterministic, so restarts would all return the
        # same model. Run once, exactly as Rust's run_model_selection does.
        models.append(run_vbhmm(data, n_states, params))
        best_model = models[-1]
        if progress_callback:
            progress_callback(n_states, params.max_hidden, best_model.lower_bound)

    # Model selection: max lower bound (ignore non-finite)
    bounds = np.array([m.lower_bound for m in models])
    finite_mask = np.isfinite(bounds)
    if finite_mask.any():
        best_idx = np.nanargmax(np.where(finite_mask, bounds, -np.inf))
    else:
        best_idx = 0
    best_n = models[best_idx].n_states

    # State assignments for all models
    state_assignments = {}
    for model in models:
        states = assign_states(model, data)
        state_assignments[model.n_states] = states

    return VBHMMResult(
        models=models,
        best_model=best_n,
        params=params,
        state_assignments=state_assignments,
    )


# ---------------------------------------------------------------------------
# 12. Export
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 12. Serialisation
#
# Split into PURE builders (no I/O) and WRITERS (the only functions in smDA
# that touch the filesystem for VBHMM output).
#
# Rationale (B0, approved 2026-08-30): the previous `update_data_csv` /
# `export_hmm_csv` wrote to a path derived from their input, so pointing an
# analysis at reference data silently destroyed it.  Keeping the writers in
# one place lets a static import-graph test assert that viewer-side modules
# never import them — see tests/test_no_viewer_writes.py.
# ---------------------------------------------------------------------------

def _write_text(text: str, output_path: str | Path, *, overwrite: bool) -> None:
    """Write *text* to *output_path*, refusing to clobber unless asked.

    Raises
    ------
    FileExistsError
        If the target exists and ``overwrite`` is False.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True to replace it. "
            f"(Refusing to overwrite implicitly: the file may be reference data.)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        f.write(text)


def write_data_csv(
    df,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a data CSV DataFrame. Refuses to clobber unless ``overwrite``."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True to replace it. "
            f"(Refusing to overwrite implicitly: the file may be reference data.)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_hmm_csv(
    text: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write hmm.csv text (normal, failure-flagged, or failure log).

    Refuses to clobber an existing file unless ``overwrite`` is True.
    """
    _write_text(text, output_path, overwrite=overwrite)


def apply_state_columns(df, result: VBHMMResult, version: str = "aas4"):
    """Return a copy of *df* with the state columns filled in.

    Pure: performs no I/O and does not mutate *df*.

    The state on row t describes the step t -> t+1, so a trajectory's final row
    has no step and carries a terminal marker instead.  The two formats spell
    both the columns and that marker differently:

        v4:  ``state(diffusion) N``   terminal marker 0
        v2:  ``Model N``              terminal marker EMPTY CELL

    ``assign_states`` already writes 0 on the final row, so for v2 those
    entries are replaced by empty strings here.  Writing v4's 0 into a v2 file
    would turn a terminal marker into state 0, which is not a state -- it
    colours as the default and reads as data.

    Values are written as strings so that a text-typed frame (see
    ``aas_format.read_data_csv_text``) round-trips byte for byte.
    """
    from smda_hmm.io import aas_format

    if version not in (aas_format.AAS2, aas_format.AAS4):
        raise ValueError(
            f"Unknown AAS version {version!r}; expected "
            f"{aas_format.AAS2!r} or {aas_format.AAS4!r}")

    out = df.copy()
    marker = aas_format.TERMINAL_MARKER[version]
    # Was the frame read as text (read_data_csv_text) or parsed as numbers?
    #
    # Tested by asking whether the first column is numeric, NOT by comparing
    # its dtype to object: pandas 3.0 gives text columns the "str" dtype
    # rather than "object", so the old test silently reported a text frame as
    # numeric.  The consequence was invisible in the reported D -- that comes
    # from the hmm.csv -- but wrote v4's numeric 0 into a v2 file's terminal
    # cells, turning the marker into what reads as state 0.
    import pandas as _pd
    is_text = (len(out.columns) > 0
               and not _pd.api.types.is_numeric_dtype(out.iloc[:, 0]))

    # Final row of each trajectory, from the trajectory id in column 0.
    traj = out.iloc[:, 0].to_numpy()
    is_last = np.empty(len(out), dtype=bool)
    if len(out):
        is_last[:-1] = traj[1:] != traj[:-1]
        is_last[-1] = True

    names = aas_format.state_column_names(
        version, result.params.min_hidden, result.params.max_hidden)
    for col_name, n_states in zip(
            names, range(result.params.min_hidden, result.params.max_hidden + 1)):
        if n_states in result.state_assignments:
            values = np.asarray(result.state_assignments[n_states])
        else:
            values = np.zeros(len(out), dtype=int)
        if is_text:
            cells = [str(int(v)) for v in values]
            for i in np.nonzero(is_last)[0]:
                cells[i] = marker
            out[col_name] = cells
        else:
            out[col_name] = values
    return out



def write_vbhmm_outputs(
    source_csv, result: VBHMMResult, *, overwrite: bool,
    data_out=None, hmm_out=None,
) -> tuple[Path, Path, str]:
    """Write the state-filled data.csv and the hmm.csv as a MATCHED PAIR.

    Returns ``(data_path, hmm_path, version)``.

    Why the two writes are one function
    -----------------------------------
    They were two calls at five call sites, and every one of them wrote the
    data.csv in the source file's version while letting the hmm.csv default to
    v4.  A v2 input therefore produced a v2 data.csv beside a v4 hmm.csv --
    a pair no AAS reader would accept, and one whose diffusion header claims
    the wrong unit.  Writing both here, from one version, removes the
    possibility rather than relying on five call sites to pass the same
    argument.

    Why the version is not selectable
    ---------------------------------
    The output version follows the input's, and cannot be chosen, because the
    two formats do not carry the same columns:

        v4 index 17-18   Mean Squared Error 1, Contours [json]
        v2 index 17      Label

    Converting either way would mean inventing the columns the target format
    has and the source does not.  Writing a fabricated per-spot error, or a
    fabricated label, into a file that reads as measured data is exactly the
    kind of thing this project forbids.  The state columns are all smDA
    computes; everything else belongs to the detector that produced the file.

    Parameters
    ----------
    source_csv : the data.csv to take the trajectory table from
    data_out : where to write it back; None writes to *source_csv* itself
    hmm_out : where to write the hmm.csv; None derives it from *data_out*
    """
    from smda_hmm.io import aas_format

    src = Path(source_csv)
    df, version = aas_format.read_data_csv_text(src)

    data_path = Path(data_out) if data_out is not None else src
    hmm_path = (Path(hmm_out) if hmm_out is not None
                else aas_format.hmm_output_path_for(data_path))

    write_data_csv(apply_state_columns(df, result, version), data_path,
                   overwrite=overwrite)
    write_hmm_csv(build_hmm_csv_text(result, version), hmm_path,
                  overwrite=overwrite)
    return data_path, hmm_path, version


def build_hmm_csv_text(result: VBHMMResult, version: str = "aas4") -> str:
    """Build hmm.csv content in AAS format. Pure: returns text, writes nothing.

    Differences between the two formats, all reproduced here:

    ==========================  =============================  ================
    item                        v4                             v2
    ==========================  =============================  ================
    first line                  JSON metadata                  ``Method,SimplevbSPT`` + blank
    ``Prior name``              1-based                        0-based
    ``wa1`` / ``wa2``           absent                         present, both ``None``
    diffusion header            ``[um^2/s]``                   ``[um/s]``
    ==========================  =============================  ================

    The v2 diffusion header carries a UNIT TYPO: the quantity is um^2/s, and
    v2 labels it um/s.  It is reproduced verbatim so that a file written here
    is byte-comparable with one written by AAS.  Correcting it would produce a
    file no v2 reader has seen, and would make round-trip comparison against
    real AAS output impossible.  The typo is documented in the README.

    v2 carries no metadata line, so timestep and distancePerPixel are NOT
    recorded in a v2 file and must be supplied by the user on the next load.

    Parameters
    ----------
    result : full VBHMM analysis result
    version : ``"aas4"`` (default) or ``"aas2"``
    """
    from datetime import datetime

    from smda_hmm.io import aas_format

    if version not in (aas_format.AAS2, aas_format.AAS4):
        raise ValueError(
            f"Unknown AAS version {version!r}; expected "
            f"{aas_format.AAS2!r} or {aas_format.AAS4!r}")
    is_v2 = version == aas_format.AAS2

    p = result.params

    lines = []

    # Line 1: JSON metadata
    meta = {
        "cTilde": p.c_tilde,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "distancePerPixel": p.distance_per_pixel,
        "estimate_mode": p.estimate_mode,
        "frameMinimum": p.frame_minimum,
        "isAddEachTrajectory": p.is_add_each_trajectory,
        "isCalcKlEach": p.is_calc_kl_each,
        "mag": p.mag,
        "maxHidden": p.max_hidden,
        "maxIter": p.max_iter,
        "method": "SimplevbSPT",
        "minHidden": p.min_hidden,
        "nTilde": p.n_tilde,
        "numRun": p.num_run,
        "software": "smda-python",
        "timestep": p.timestep,
        "type": "hmm",
        "version": "1.0",
        "wBTilde": p.w_b_tilde,
        "wPiTilde": p.w_pi_tilde,
    }
    if is_v2:
        # v2 has no metadata line; it opens with the method name and a blank.
        lines.append("Method,SimplevbSPT")
        lines.append("")
    else:
        lines.append(json.dumps(meta))

    for model in result.models:
        N = model.n_states
        st = model.state

        # Sort by D ascending
        d_order = np.argsort(model.D)

        lines.append(f"Model,{N}")
        lines.append("Name,Lower bound,lnZs,kl,kl pi, kl diffusion, kl b")
        lines.append(
            f"Value,{model.lower_bound:.2f},{model.ln_zs:.2f},"
            f"{model.kl},{model.kl_pi}, {model.kl_diffusion}, {model.kl_b}"
        )

        # Prior names: v4 numbers the states from 1, v2 from 0.
        first = 0 if is_v2 else 1
        lines.append("Prior name,"
                     + ",".join(str(i + first) for i in range(N)))
        if is_v2:
            # v2 emits these two placeholder rows; AAS writes None in both.
            lines.append("wa1,None")
            lines.append("wa2,None")

        # wpi (sorted)
        wpi_sorted = st.w_pi[d_order]
        lines.append("wpi," + ",".join(f"{v}" for v in wpi_sorted))

        # n (sorted)
        n_sorted = st.n[d_order]
        lines.append("n," + ",".join(f"{v}" for v in n_sorted))

        # c (sorted)
        c_sorted = st.c[d_order]
        lines.append("c," + ",".join(f"{v}" for v in c_sorted))

        # wB (sorted rows and columns)
        for i in range(N):
            row = st.w_b[d_order[i], d_order]
            lines.append(f"wb[{i}]," + ",".join(f"{v}" for v in row))

        # Diffusion coefficients
        D_sorted = model.D[d_order]
        dt = result.params.timestep

        # D std from posterior: Var[γ] = n/c², D ≈ c/(4*dt*n),
        # delta method: std_D ≈ D / sqrt(n)  (approximate)
        D_std = D_sorted / np.sqrt(n_sorted)

        # Initial probability
        wpi_sum = np.sum(wpi_sorted)
        init_prob = wpi_sorted / wpi_sum
        # Std from Dirichlet: std = sqrt(a_i(a_0-a_i)/(a_0²(a_0+1)))
        init_std = np.sqrt(
            wpi_sorted * (wpi_sum - wpi_sorted) / (wpi_sum**2 * (wpi_sum + 1))
        )

        # "[um/s]" in v2 is a unit typo in the format itself (the quantity is
        # um^2/s).  Reproduced deliberately -- see this function's docstring.
        _unit = "[um/s]" if is_v2 else "[um^2/s]"
        lines.append(f",Diffusion coefficient{_unit},,Initial Probability,")
        lines.append("State,Average,Std,Average,Std")
        for k in range(N):
            lines.append(
                f"{k + 1},{D_sorted[k]},{D_std[k]},{init_prob[k]},{init_std[k]}"
            )

        # Transition probability
        header_parts = []
        for k in range(N):
            header_parts.extend([f"{k + 1}(Ave)", f"{k + 1}(Std)"])
        lines.append("Transition Probability," + ",".join(header_parts))

        wB_sorted = st.w_b[np.ix_(d_order, d_order)]
        for i in range(N):
            row_sum = np.sum(wB_sorted[i])
            trans_prob = wB_sorted[i] / row_sum
            trans_std = np.sqrt(
                wB_sorted[i] * (row_sum - wB_sorted[i])
                / (row_sum**2 * (row_sum + 1))
            )
            parts = [str(i + 1)]
            for k in range(N):
                parts.extend([f"{trans_prob[k]}", f"{trans_std[k]}"])
            lines.append(",".join(parts))

        lines.append("")

    # Suitable model
    lines.append(f"Suitable Model,{result.best_model}")

    # Convergence info (appended after Suitable Model for AAS4 compatibility)
    for model in result.models:
        lines.append(
            f"Converged,{model.n_states},{model.converged},{model.n_iter}")

    return "\n".join(lines) + "\n"


def build_failed_hmm_csv_text(
    params: VBHMMParams,
    error: str,
    traceback_str: str = "",
) -> str:
    """Build a failure-flagged hmm.csv. Pure: returns text, writes nothing.

    The JSON metadata line contains ``"failure": true`` so that
    ``load_vbhmm_from_csv`` can detect it and return an empty result.
    The file otherwise mirrors the normal hmm.csv structure with NaN
    in all numeric fields.
    """
    from datetime import datetime
    from dataclasses import asdict

    meta = {
        "failure": True,
        "failure_reason": str(error),
        "failure_traceback": traceback_str[:2000],
        "failure_timestamp": datetime.now().isoformat(),
        "params_snapshot": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in asdict(params).items()
        },
        "timestep": params.timestep,
        "minHidden": params.min_hidden,
        "maxHidden": params.max_hidden,
        "software": "smda-python",
        "type": "hmm",
    }

    lines = [json.dumps(meta)]

    for n in range(params.min_hidden, params.max_hidden + 1):
        lines.append(f"Model,{n}")
        lines.append("Name,Lower bound,lnZs,kl,kl pi, kl diffusion, kl b")
        lines.append("Value,nan,nan,nan,nan,nan,nan")
        lines.append("Prior name," + ",".join(str(i + 1) for i in range(n)))
        lines.append("wpi," + ",".join("nan" for _ in range(n)))
        lines.append("n," + ",".join("nan" for _ in range(n)))
        lines.append("c," + ",".join("nan" for _ in range(n)))
        for i in range(n):
            lines.append(f"wb[{i}]," + ",".join("nan" for _ in range(n)))
        lines.append(",Diffusion coefficient[um^2/s],,Initial Probability,")
        lines.append("State,Average,Std,Average,Std")
        for k in range(n):
            lines.append(f"{k + 1},nan,nan,nan,nan")
        header_parts = []
        for k in range(n):
            header_parts.extend([f"{k + 1}(Ave)", f"{k + 1}(Std)"])
        lines.append("Transition Probability," + ",".join(header_parts))
        for i in range(n):
            parts = [str(i + 1)]
            for k in range(n):
                parts.extend(["nan", "nan"])
            lines.append(",".join(parts))
        lines.append(f"Converged,False,0")
        lines.append("")

    lines.append("Suitable Model,0")

    return "\n".join(lines) + "\n"


def build_vbhmm_failure_log_text(
    basename: str,
    error: Exception,
    traceback_str: str,
    params: VBHMMParams,
) -> str:
    """Build the {basename}_hmm_FAILED.csv log. Pure: returns text."""
    from datetime import datetime
    from dataclasses import asdict

    lines = [
        f"timestamp,{datetime.now().isoformat()}",
        f"basename,{basename}",
        f"exception_class,{type(error).__name__}",
        f"message,\"{str(error)}\"",
        "",
        "traceback",
    ]
    lines.extend(traceback_str.strip().split("\n"))
    lines.append("")
    lines.append("params")
    for k, v in asdict(params).items():
        lines.append(f"{k},{v}")

    return "\n".join(lines) + "\n"


def load_vbhmm_from_csv(
    hmm_path: str | Path,
    data_path: str | Path,
) -> VBHMMResult:
    """Restore VBHMMResult from *_hmm.csv + *_data.csv on disk.

    All fields referenced by _render_vbhmm_results() are restored from
    the CSV files.  No dummy values are used for VBHMMState fields.

    Parameters
    ----------
    hmm_path : path to hmm.csv (AAS or smda-python format)
    data_path : path to data.csv (with state(diffusion) columns)

    Returns
    -------
    VBHMMResult with all models, state assignments, and params.

    Raises
    ------
    ValueError
        If files are missing, malformed, or have missing columns/fields.
    """
    import pandas as pd

    hmm_path = Path(hmm_path)
    data_path = Path(data_path)

    if not hmm_path.exists():
        raise ValueError(f"Missing file: {hmm_path}")
    if not data_path.exists():
        raise ValueError(f"Missing file: {data_path}")

    with open(hmm_path, "r") as f:
        lines = [line.rstrip("\n").rstrip("\r") for line in f.readlines()]

    if not lines:
        raise ValueError(f"Empty hmm.csv: {hmm_path}")

    # --- Line 1: JSON metadata → VBHMMParams ---
    raw_meta = lines[0].strip().strip('"').replace('""', '"')
    try:
        meta = json.loads(raw_meta)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Malformed hmm.csv: invalid JSON metadata in {hmm_path}: {e}"
        ) from e

    # --- Failure flag: return empty result for failed cells ---
    if meta.get("failure", False):
        _min_h = int(meta.get("minHidden", 1))
        _max_h = int(meta.get("maxHidden", 5))
        _ts = float(meta.get("timestep", -1.0))
        _fail_params = VBHMMParams(
            min_hidden=_min_h, max_hidden=_max_h, timestep=_ts,
        )
        return VBHMMResult(
            models=[], best_model=0,
            params=_fail_params, state_assignments={},
        )

    _required_meta = ["minHidden", "maxHidden", "timestep"]
    for key in _required_meta:
        if key not in meta:
            raise ValueError(
                f"Malformed hmm.csv: missing metadata key '{key}' in {hmm_path}"
            )

    params = VBHMMParams(
        n_tilde=meta.get("nTilde", 1.0),
        c_tilde=meta.get("cTilde", 0.001),
        w_pi_tilde=meta.get("wPiTilde", 1.0),
        w_b_tilde=meta.get("wBTilde", 0.01),
        mag=meta.get("mag", 30.0),
        max_hidden=int(meta["maxHidden"]),
        min_hidden=int(meta["minHidden"]),
        max_iter=int(meta.get("maxIter", 100)),
        num_run=int(meta.get("numRun", 5)),
        timestep=float(meta["timestep"]),
        distance_per_pixel=float(meta.get("distancePerPixel", -1.0)),
        frame_minimum=int(meta.get("frameMinimum", 1)),
        estimate_mode=str(meta.get("estimate_mode", "MaxProb")),
        is_add_each_trajectory=bool(meta.get("isAddEachTrajectory", True)),
        is_calc_kl_each=bool(meta.get("isCalcKlEach", False)),
    )

    # --- Parse model sections ---
    models: list[VBHMMModelResult] = []
    best_model: int | None = None
    i = 1

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Suitable Model"):
            parts = line.split(",")
            best_model = int(parts[1].strip())
            i += 1
            continue

        if not line.startswith("Model,"):
            i += 1
            continue

        n_states = int(line.split(",")[1].strip())
        ctx = f"Model {n_states} in {hmm_path}"

        # Value line (skip header)
        i += 1  # "Name,Lower bound,..."
        i += 1  # "Value,..."
        if i >= len(lines):
            raise ValueError(f"Malformed hmm.csv: truncated {ctx}")
        vals = lines[i].strip().split(",")
        if len(vals) < 7:
            raise ValueError(
                f"Malformed hmm.csv: Value line has {len(vals)} fields "
                f"(expected 7) in {ctx}"
            )
        lower_bound = float(vals[1])
        ln_zs = float(vals[2])
        kl = float(vals[3])
        kl_pi = float(vals[4])
        kl_diffusion = float(vals[5])
        kl_b = float(vals[6])

        # Prior name
        i += 1

        # wpi
        i += 1
        if i >= len(lines):
            raise ValueError(f"Malformed hmm.csv: missing 'wpi' in {ctx}")
        wpi_line = lines[i].strip().split(",")
        if not wpi_line[0].strip().startswith("wpi"):
            raise ValueError(
                f"Malformed hmm.csv: expected 'wpi' line, got "
                f"'{wpi_line[0]}' in {ctx}"
            )
        w_pi = np.array([float(x) for x in wpi_line[1:] if x.strip()])

        # n
        i += 1
        if i >= len(lines):
            raise ValueError(f"Malformed hmm.csv: missing 'n' in {ctx}")
        n_line = lines[i].strip().split(",")
        if not n_line[0].strip() == "n":
            raise ValueError(
                f"Malformed hmm.csv: expected 'n' line, got "
                f"'{n_line[0]}' in {ctx}"
            )
        state_n = np.array([float(x) for x in n_line[1:] if x.strip()])

        # c
        i += 1
        if i >= len(lines):
            raise ValueError(f"Malformed hmm.csv: missing 'c' in {ctx}")
        c_line = lines[i].strip().split(",")
        if not c_line[0].strip() == "c":
            raise ValueError(
                f"Malformed hmm.csv: expected 'c' line, got "
                f"'{c_line[0]}' in {ctx}"
            )
        state_c = np.array([float(x) for x in c_line[1:] if x.strip()])

        # wb rows
        w_b = np.zeros((n_states, n_states))
        for row_idx in range(n_states):
            i += 1
            if i >= len(lines):
                raise ValueError(
                    f"Malformed hmm.csv: missing 'wb[{row_idx}]' in {ctx}"
                )
            wb_line = lines[i].strip().split(",")
            expected_prefix = f"wb[{row_idx}]"
            if not wb_line[0].strip().startswith(expected_prefix):
                raise ValueError(
                    f"Malformed hmm.csv: expected '{expected_prefix}', got "
                    f"'{wb_line[0]}' in {ctx}"
                )
            w_b[row_idx] = [float(x) for x in wb_line[1:] if x.strip()]

        # D values (from State lines after "State,Average,Std,..." header)
        D = np.zeros(n_states)
        i += 1  # ",Diffusion coefficient..."
        i += 1  # "State,Average,Std,Average,Std"
        for k in range(n_states):
            i += 1
            if i >= len(lines):
                raise ValueError(
                    f"Malformed hmm.csv: missing D for state {k+1} in {ctx}"
                )
            parts = lines[i].strip().split(",")
            D[k] = float(parts[1])

        # Skip Transition Probability section
        i += 1  # "Transition Probability,..."
        for _ in range(n_states):
            i += 1

        # Converged line (optional — smda-python output only, not in AAS)
        converged = True
        n_iter = 0
        i += 1
        if i < len(lines):
            cline = lines[i].strip()
            if cline.startswith("Converged"):
                cparts = cline.split(",")
                converged = cparts[1].strip() == "True"
                n_iter = int(cparts[2].strip()) if len(cparts) > 2 else 0
                i += 1  # advance past Converged line

        # hmm.csv stores values already D-sorted. np.argsort(D) on
        # sorted D returns identity permutation, so _render_vbhmm_results
        # will not re-sort.
        state = VBHMMState(n=state_n, c=state_c, w_pi=w_pi, w_b=w_b)
        priors = VBHMMPriors(
            n=np.zeros(n_states), c=np.zeros(n_states),
            w_pi=np.zeros(n_states), w_b=np.zeros((n_states, n_states)),
        )
        pst = np.empty((0, n_states))

        model = VBHMMModelResult(
            n_states=n_states,
            lower_bound=lower_bound,
            ln_zs=ln_zs,
            kl=kl, kl_pi=kl_pi, kl_diffusion=kl_diffusion, kl_b=kl_b,
            state=state,
            priors=priors,
            pst=pst,
            converged=converged,
            n_iter=n_iter,
            D=D,
        )
        models.append(model)

        # Skip blank line between models
        if i < len(lines) and not lines[i].strip():
            i += 1

    if best_model is None:
        raise ValueError(
            f"Malformed hmm.csv: 'Suitable Model' line not found in {hmm_path}"
        )
    if not models:
        raise ValueError(
            f"Malformed hmm.csv: no Model sections found in {hmm_path}"
        )

    # --- Parse state assignments from data.csv ---
    df = pd.read_csv(data_path)
    state_assignments: dict[int, np.ndarray] = {}
    for n_st in range(params.min_hidden, params.max_hidden + 1):
        # Either spelling: v4 "state(diffusion) N", v2 "Model N".
        col = aas_format.find_state_column(df.columns, n_st)
        if col is None:
            raise ValueError(
                f"No {n_st}-state column in {data_path}. Expected "
                f"'state(diffusion) {n_st}' (AAS v4) or 'Model {n_st}' "
                f"(AAS v2)."
            )
        state_assignments[n_st] = df[col].values.astype(np.int64)

    result = VBHMMResult(
        models=models,
        best_model=best_model,
        params=params,
        state_assignments=state_assignments,
    )

    # --- Integrity assertion: all fields _render_vbhmm_results needs ---
    for idx, m in enumerate(result.models):
        assert m.D is not None and len(m.D) == m.n_states, (
            f"Model N={m.n_states}: D array missing or wrong size"
        )
        assert m.state.w_b is not None and m.state.w_b.shape == (m.n_states, m.n_states), (
            f"Model N={m.n_states}: w_b missing or wrong shape"
        )
        assert m.state.w_pi is not None and len(m.state.w_pi) == m.n_states, (
            f"Model N={m.n_states}: w_pi missing or wrong size"
        )
        assert m.state.n is not None and len(m.state.n) == m.n_states, (
            f"Model N={m.n_states}: state.n missing or wrong size"
        )

    return result
