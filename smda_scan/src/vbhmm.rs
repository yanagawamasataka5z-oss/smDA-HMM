//! Variational Bayesian HMM (vbSPT) — faithful port of smda/core/vbhmm.py.
//!
//! Key structures:
//!   dx2[t]           — concatenated squared displacements (μm²)
//!   trj_starts[m]    — first step index of trajectory m
//!   trj_ends[m]      — last step index of trajectory m (inclusive)
//!   N                — number of hidden states
//!
//! Parallel: Forward-Backward is parallelized over trajectories via rayon.

use std::f64::consts::PI;
use statrs::function::gamma::{digamma, ln_gamma};

// =========================================================================
// Data types
// =========================================================================

/// Spatial dimension of the diffusion model.
///
/// This is NOT a parameter.  `preprocess_trajectories` collapses the data to
/// dx2 = dx^2 + dy^2 before it ever reaches this module, so the model is 2D by
/// construction.  Exposing a `dim` knob would let a caller request 3D and get a
/// silently wrong answer.  AAS agrees: neither its hmm.csv metadata nor
/// settings.csv carries a dimension field (settings.csv is headed
/// "Tracking2DSettings").
const SPATIAL_DIM: usize = 2;

/// Relative-change threshold on the convergence bound.
///
/// Must stay identical to smda/core/vbhmm.py:VBEM_CONV_TOL, where the choice
/// is documented — the two implementations are verified against each other
/// down to the iteration count.  Raised from 1e-8 on 2026-08-31 together with
/// the fix that made the monitored quantity monotone; the value is the
/// tightest that still fires inside max_iter with margin at the selected
/// model, and is set from our own iteration, not from AAS.
const VBEM_CONV_TOL: f64 = 1e-12;

#[derive(Clone)]
pub struct VBHMMParams {
    pub n_tilde: f64,
    pub c_tilde: f64,
    pub w_pi_tilde: f64,
    pub w_b_tilde: f64,
    pub mag: f64,
    pub max_iter: usize,
    pub num_run: usize,
    pub timestep: f64,
    pub w_pi_scale_mode: u8, // 0: unscaled, 1: scaled by n_traj (default: 1)
    pub w_b_scale_mode: u8,  // 0: unscaled, 1: scaled (off-diag + diag by mag, AAS P5)
    pub is_calc_kl_each: bool, // true: use scaled priors for KL, false: use unscaled base priors
    pub is_add_each_trajectory: bool, // true: scale priors by trajectory count M
}

#[derive(Clone)]
pub struct Priors {
    pub n: Vec<f64>,
    pub c: Vec<f64>,
    pub w_pi: Vec<f64>,
    pub w_b: Vec<f64>, // N×N row-major
}

#[derive(Clone)]
pub struct State {
    pub n: Vec<f64>,
    pub c: Vec<f64>,
    pub w_pi: Vec<f64>,
    pub w_b: Vec<f64>, // N×N row-major
}

pub struct VBHMMSingleResult {
    pub lower_bound: f64,
    pub ln_zs: f64,
    pub kl: f64,
    pub kl_pi: f64,
    pub kl_diffusion: f64,
    pub kl_b: f64,
    pub state: State,
    pub pst: Vec<f64>, // T×N row-major
    pub d_values: Vec<f64>,
    pub converged: bool,
    pub n_iter: usize,
}

pub struct VBHMMFullResult {
    pub models: Vec<VBHMMSingleResult>,
    pub best_n: usize,
}

// =========================================================================
// Priors
// =========================================================================

fn build_priors(n_states: usize, n_traj: usize, p: &VBHMMParams, scale: bool) -> Priors {
    // M = trajectory count when isAddEachTrajectory is on, otherwise 1.
    // Mirrors smda/core/vbhmm.py:_build_priors.
    let m = if p.is_add_each_trajectory { n_traj as f64 } else { 1.0 };
    // n, c: scaled by M when scale=true
    let s = if scale { m } else { 1.0 };
    let n = vec![p.n_tilde * s; n_states];
    let c = vec![p.c_tilde * s; n_states];

    // w_pi: mode 0 = unscaled (wPiTilde), mode 1 = scaled (wPiTilde × n_traj)
    let s_pi = if p.w_pi_scale_mode == 1 && scale { m } else { 1.0 };
    let w_pi = vec![p.w_pi_tilde * s_pi; n_states];

    // w_b: mode 0 = unscaled, mode 1 = scaled (AAS P5)
    let s_b = if p.w_b_scale_mode == 1 && scale { m } else { 1.0 };
    let mut w_b = vec![0.0; n_states * n_states];
    for i in 0..n_states {
        for j in 0..n_states {
            if i != j {
                w_b[i * n_states + j] = p.w_b_tilde * s_b;
            }
        }
    }
    // mode 1: diagonal = mag × wBTilde × n_traj (AAS P5)
    if p.w_b_scale_mode == 1 && scale {
        for i in 0..n_states {
            w_b[i * n_states + i] = p.mag * p.w_b_tilde * m;
        }
    }
    Priors { n, c, w_pi, w_b }
}

// =========================================================================
// K-means++ initialization
//
// Deterministic K-means++ (seedless variant):
// 1. First centroid = median of data along each dimension
// 2. Subsequent centroids = farthest point from existing centroids
// This guarantees reproducibility without requiring a random seed.
// =========================================================================

fn kmeans_init(
    dx2: &[f64],
    trj_starts: &[usize],
    trj_ends: &[usize],
    n_states: usize,
    priors: &Priors,
    params: &VBHMMParams,
) -> State {
    let t = dx2.len();
    if t == 0 || n_states == 0 {
        return State {
            n: priors.n.clone(), c: priors.c.clone(),
            w_pi: priors.w_pi.clone(), w_b: priors.w_b.clone(),
        };
    }

    let dim2 = SPATIAL_DIM as f64 / 2.0; // = 1.0 for 2D
    let n = n_states;

    // log(dx2) for k-means
    let log_dx2: Vec<f64> = dx2.iter().map(|&v| (v.max(1e-300)).ln()).collect();

    // Deterministic K-means++ center selection on log_dx2
    // First centroid: median of log_dx2 (deterministic, picks "typical" diffusion)
    let mut centers = Vec::with_capacity(n);
    {
        let mut sorted_log = log_dx2.clone();
        sorted_log.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_val = sorted_log[sorted_log.len() / 2];
        // Pick the data point closest to the median
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, &v) in log_dx2.iter().enumerate() {
            let d = (v - median_val).abs();
            if d < best_dist { best_dist = d; best_idx = i; }
        }
        centers.push(log_dx2[best_idx]);
    }

    // Subsequent centroids: farthest-point heuristic (max D² distance)
    for _ in 1..n {
        let mut best_idx = 0;
        let mut best_d2 = -1.0_f64;
        for (i, &v) in log_dx2.iter().enumerate() {
            let min_d2 = centers.iter()
                .map(|&c| (v - c) * (v - c))
                .fold(f64::MAX, f64::min);
            if min_d2 > best_d2 { best_d2 = min_d2; best_idx = i; }
        }
        centers.push(log_dx2[best_idx]);
    }

    // Lloyd's algorithm (50 iterations on linear dx2)
    // Convert centers from log to linear
    let mut lin_centers: Vec<f64> = centers.iter().map(|&c| c.exp()).collect();
    let mut labels = vec![0usize; t];

    for _ in 0..50 {
        // Assign
        for i in 0..t {
            let mut best_k = 0;
            let mut best_d = f64::MAX;
            for k in 0..n {
                let d = (dx2[i] - lin_centers[k]).abs();
                if d < best_d { best_d = d; best_k = k; }
            }
            labels[i] = best_k;
        }
        // Update
        let mut sums = vec![0.0; n];
        let mut counts = vec![0usize; n];
        for i in 0..t {
            sums[labels[i]] += dx2[i];
            counts[labels[i]] += 1;
        }
        let mut changed = false;
        for k in 0..n {
            let new_c = if counts[k] > 0 { sums[k] / counts[k] as f64 } else { lin_centers[k] };
            if (new_c - lin_centers[k]).abs() > 1e-10 { changed = true; }
            lin_centers[k] = new_c;
        }
        if !changed { break; }
    }

    // Re-assign final labels
    for i in 0..t {
        let mut best_k = 0;
        let mut best_d = f64::MAX;
        for k in 0..n {
            let d = (dx2[i] - lin_centers[k]).abs();
            if d < best_d { best_d = d; best_k = k; }
        }
        labels[i] = best_k;
    }

    // Sort clusters by mean dx2 ascending
    let mut cluster_means: Vec<(f64, usize)> = (0..n).map(|k| {
        let sum: f64 = (0..t).filter(|&i| labels[i] == k).map(|i| dx2[i]).sum();
        let cnt = (0..t).filter(|&i| labels[i] == k).count();
        let mean = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
        (mean, k)
    }).collect();
    cluster_means.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut old_to_new = vec![0usize; n];
    for (new_k, &(_, old_k)) in cluster_means.iter().enumerate() {
        old_to_new[old_k] = new_k;
    }
    for i in 0..t {
        labels[i] = old_to_new[labels[i]];
    }

    // Build initial state
    let mut n_init = priors.n.clone();
    let mut c_init = priors.c.clone();
    for k in 0..n {
        let cnt: usize = (0..t).filter(|&i| labels[i] == k).count();
        let sum_dx2: f64 = (0..t).filter(|&i| labels[i] == k).map(|i| dx2[i]).sum();
        n_init[k] += dim2 * cnt as f64;
        c_init[k] += sum_dx2;
    }

    // Initial w_pi from trajectory start labels
    let mut w_pi_init = priors.w_pi.clone();
    for &s in trj_starts {
        if s < t {
            w_pi_init[labels[s]] += 1.0;
        }
    }

    // Initial w_b from transition counts
    let mut w_b_init = priors.w_b.clone();
    let m = trj_starts.len();
    for mi in 0..m {
        let start = trj_starts[mi];
        let end = trj_ends[mi];
        for ti in start..end {
            let from = labels[ti];
            let to = labels[ti + 1];
            let idx = from * n + to;
            if from == to {
                w_b_init[idx] += params.mag * 1.0;
            } else {
                w_b_init[idx] += 1.0;
            }
        }
    }
    // Ensure diagonal has at least mag * 1.0
    for i in 0..n {
        let idx = i * n + i;
        if w_b_init[idx] < params.mag {
            w_b_init[idx] = params.mag;
        }
    }

    State { n: n_init, c: c_init, w_pi: w_pi_init, w_b: w_b_init }
}

// =========================================================================
// Forward-Backward (single trajectory)
// =========================================================================

fn fb_single(
    q: &[f64],   // N×N transition probs (pre-normalized)
    h: &[f64],   // T_m×N emission probs (pre-normalized)
    t_len: usize,
    n: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    // Returns (lnZ, wA_flat[N×N], pst_flat[T_m×N])
    let mut alpha = vec![0.0; t_len * n];
    let mut scale = vec![0.0; t_len];

    // Forward
    let mut s = 0.0;
    for j in 0..n {
        alpha[j] = h[j];
        s += alpha[j];
    }
    if s > 0.0 {
        for j in 0..n { alpha[j] /= s; }
        scale[0] = s;
    } else {
        for j in 0..n { alpha[j] = 1.0 / n as f64; }
        scale[0] = 1e-300;
    }

    for t in 1..t_len {
        let mut st = 0.0;
        for j in 0..n {
            let mut sum_ij = 0.0;
            for i in 0..n {
                sum_ij += alpha[(t - 1) * n + i] * q[i * n + j];
            }
            alpha[t * n + j] = sum_ij * h[t * n + j];
            st += alpha[t * n + j];
        }
        if st > 0.0 {
            for j in 0..n { alpha[t * n + j] /= st; }
            scale[t] = st;
        } else {
            for j in 0..n { alpha[t * n + j] = 1.0 / n as f64; }
            scale[t] = 1e-300;
        }
    }

    // Backward
    let mut beta = vec![0.0; t_len * n];
    for j in 0..n { beta[(t_len - 1) * n + j] = 1.0; }

    for t in (0..t_len - 1).rev() {
        let mut sb = 0.0;
        for i in 0..n {
            let mut val = 0.0;
            for j in 0..n {
                val += q[i * n + j] * h[(t + 1) * n + j] * beta[(t + 1) * n + j];
            }
            beta[t * n + i] = val;
            sb += val;
        }
        if sb > 0.0 {
            for i in 0..n { beta[t * n + i] /= sb; }
        }
    }

    // Posterior
    let mut pst = vec![0.0; t_len * n];
    for t in 0..t_len {
        let mut s = 0.0;
        for j in 0..n {
            pst[t * n + j] = alpha[t * n + j] * beta[t * n + j];
            s += pst[t * n + j];
        }
        if s > 0.0 {
            for j in 0..n { pst[t * n + j] /= s; }
        }
    }

    // Transition counts (xi): compute normalized directly, no undo/redo
    let mut wa = vec![0.0; n * n];
    for t in 0..t_len - 1 {
        // Compute normalization factor first
        let mut norm = 0.0;
        for i in 0..n {
            for j in 0..n {
                norm += alpha[t * n + i] * q[i * n + j]
                    * h[(t + 1) * n + j] * beta[(t + 1) * n + j];
            }
        }
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        // Accumulate normalized xi
        for i in 0..n {
            for j in 0..n {
                wa[i * n + j] += alpha[t * n + i] * q[i * n + j]
                    * h[(t + 1) * n + j] * beta[(t + 1) * n + j] * inv_norm;
            }
        }
    }

    // lnZ
    let lnz: f64 = scale.iter().map(|&s| s.max(1e-300).ln()).sum();

    (lnz, wa, pst)
}

// =========================================================================
// Forward-Backward (all trajectories, parallel)
// =========================================================================

fn forward_backward_all(
    ln_h: &[f64],   // T×N
    ln_q: &[f64],   // N×N
    trj_starts: &[usize],
    trj_ends: &[usize],
    t_total: usize,
    n: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    use rayon::prelude::*;

    // Pre-normalize Q and H (matching Python)
    let ln_q_max = ln_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let q_norm: Vec<f64> = ln_q.iter().map(|&v| (v - ln_q_max).exp()).collect();
    let mut ln_h_max = vec![f64::NEG_INFINITY; t_total];
    for t in 0..t_total {
        for j in 0..n {
            let v = ln_h[t * n + j];
            if v > ln_h_max[t] { ln_h_max[t] = v; }
        }
    }
    let mut h_norm = vec![0.0; t_total * n];
    for t in 0..t_total {
        for j in 0..n {
            h_norm[t * n + j] = (ln_h[t * n + j] - ln_h_max[t]).exp();
        }
    }

    let m = trj_starts.len();

    // Parallel over trajectories
    let per_trj: Vec<(usize, usize, f64, Vec<f64>, Vec<f64>)> = (0..m)
        .into_par_iter()
        .map(|mi| {
            let start = trj_starts[mi];
            let end = trj_ends[mi]; // inclusive
            let t_len = end - start + 1;

            // H for this trajectory: contiguous slice (start*n .. (end+1)*n)
            let h_trj = &h_norm[start * n .. (start + t_len) * n];

            let (lnz, wa, pst) = fb_single(&q_norm, h_trj, t_len, n);
            (start, t_len, lnz, wa, pst)
        })
        .collect();

    // Aggregate
    let mut pst_all = vec![0.0; t_total * n];
    let mut wa_all = vec![0.0; n * n];
    let mut lnz_total = 0.0;

    for (start, t_len, lnz, wa, pst) in per_trj {
        lnz_total += lnz;
        for i in 0..n * n { wa_all[i] += wa[i]; }
        for t in 0..t_len {
            for j in 0..n {
                pst_all[(start + t) * n + j] = pst[t * n + j];
            }
        }
    }

    (lnz_total, wa_all, pst_all)
}

// =========================================================================
// E-step
// =========================================================================

fn e_step(
    dx2: &[f64],
    state: &State,
    trj_starts: &[usize],
    trj_ends: &[usize],
    dim: usize,
) -> (Vec<f64>, Vec<f64>, f64, f64, f64) {
    // Returns (pst, wA, lnZz, lnZQ, lnZq)
    let t = dx2.len();
    let n = state.n.len();
    let m = trj_starts.len();
    let dim2 = dim as f64 / 2.0;

    // Emission log-probabilities
    let mut ln_h = vec![0.0; t * n];
    let mut ln_h0 = vec![0.0; n];
    let mut gamma_rate = vec![0.0; n];
    for j in 0..n {
        ln_h0[j] = dim2 * (digamma(state.n[j]) - (PI * state.c[j]).ln());
        gamma_rate[j] = state.n[j] / state.c[j];
    }
    for ti in 0..t {
        for j in 0..n {
            ln_h[ti * n + j] = ln_h0[j] - gamma_rate[j] * dx2[ti];
        }
    }

    // Add initial state log-probability at trajectory starts
    let w_pi_sum: f64 = state.w_pi.iter().sum();
    let mut ln_pi = vec![0.0; n];
    for j in 0..n {
        ln_pi[j] = digamma(state.w_pi[j]) - digamma(w_pi_sum);
    }
    for mi in 0..m {
        let s = trj_starts[mi];
        for j in 0..n {
            ln_h[s * n + j] += ln_pi[j];
        }
    }

    // Transition log-probabilities
    let mut ln_q = vec![0.0; n * n];
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| state.w_b[i * n + j]).sum();
        for j in 0..n {
            ln_q[i * n + j] = digamma(state.w_b[i * n + j]) - digamma(row_sum);
        }
    }

    // lnZQ, lnZq for lower bound
    let ln_q_max = ln_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let total_trans = t as f64 - m as f64;
    let ln_zq_big = total_trans * ln_q_max;

    let mut ln_h_max_sum = 0.0;
    for ti in 0..t {
        let mut mx = f64::NEG_INFINITY;
        for j in 0..n {
            if ln_h[ti * n + j] > mx { mx = ln_h[ti * n + j]; }
        }
        ln_h_max_sum += mx;
    }

    // Forward-Backward
    let (lnzz, wa, pst) = forward_backward_all(&ln_h, &ln_q, trj_starts, trj_ends, t, n);

    (pst, wa, lnzz, ln_zq_big, ln_h_max_sum)
}

// =========================================================================
// M-step
// =========================================================================

fn m_step(
    dx2: &[f64],
    pst: &[f64],
    wa: &[f64],
    priors: &Priors,
    trj_starts: &[usize],
    dim: usize,
) -> State {
    let t = dx2.len();
    let n = priors.n.len();
    let dim2 = dim as f64 / 2.0;

    // w_pi
    let mut w_pi = priors.w_pi.clone();
    for &s in trj_starts {
        for j in 0..n {
            w_pi[j] += pst[s * n + j];
        }
    }

    // w_b
    let mut w_b = priors.w_b.clone();
    for i in 0..n * n {
        w_b[i] += wa[i];
    }

    // n, c
    let mut pst_sum = vec![0.0; n];
    let mut pst_dx2 = vec![0.0; n];
    for ti in 0..t {
        for j in 0..n {
            pst_sum[j] += pst[ti * n + j];
            pst_dx2[j] += pst[ti * n + j] * dx2[ti];
        }
    }
    let mut n_new = priors.n.clone();
    let mut c_new = priors.c.clone();
    for j in 0..n {
        n_new[j] += dim2 * pst_sum[j];
        c_new[j] += pst_dx2[j];
    }

    State { n: n_new, c: c_new, w_pi, w_b }
}

// =========================================================================
// KL divergence terms
// =========================================================================

fn kl_dirichlet(w: &[f64], u: &[f64]) -> f64 {
    // Clamp Dirichlet parameters to eps to avoid ln_gamma/digamma singularities.
    // When w_b has zero entries (unused transitions in N=5), digamma(~0) → -inf.
    // eps=0.01 matches the scale of w_b_tilde (prior off-diagonal = 0.01).
    let eps = 0.01;
    let n = w.len();
    let w0: f64 = w.iter().map(|&v| v.max(eps)).sum();
    let u0: f64 = u.iter().map(|&v| v.max(eps)).sum();
    let mut kl = ln_gamma(w0) - ln_gamma(u0);
    let psi_w0 = digamma(w0);
    for i in 0..n {
        let wi = w[i].max(eps);
        let ui = u[i].max(eps);
        kl += ln_gamma(ui) - ln_gamma(wi);
        kl += (wi - ui) * (digamma(wi) - psi_w0);
    }
    if kl.is_finite() { kl.max(0.0) } else { 0.0 }
}

fn kl_diffusion(state_n: &[f64], state_c: &[f64], prior_n: &[f64], prior_c: &[f64]) -> f64 {
    let n = state_n.len();
    let mut kl = 0.0;
    for j in 0..n {
        let kl_j = prior_n[j] * (state_c[j] / prior_c[j]).ln()
            - state_n[j] * (1.0 - prior_c[j] / state_c[j])
            - ln_gamma(state_n[j]) + ln_gamma(prior_n[j])
            + (state_n[j] - prior_n[j]) * digamma(state_n[j]);
        if kl_j.is_finite() { kl += kl_j; }
    }
    kl
}

fn compute_lower_bound(
    ln_zq: f64, ln_zq2: f64, ln_zz: f64,
    state: &State, base: &Priors, mag: f64,
) -> (f64, f64, f64, f64, f64) {
    let n = state.n.len();

    let kl_pi = kl_dirichlet(&state.w_pi, &base.w_pi);
    let kl_diff = kl_diffusion(&state.n, &state.c, &base.n, &base.c);

    let kl_b = if n <= 1 {
        0.0
    } else {
        // Build full prior w_b: off-diagonal = wBTilde, diagonal = wBTilde * mag
        // AAS uses mag-scaled diagonal for KL prior (verified numerically in Phase 4 audit)
        let off_diag = base.w_b[0 * n + 1]; // off-diagonal value = wBTilde
        let mut prior_full = base.w_b.clone();
        for i in 0..n { prior_full[i * n + i] = off_diag * mag; }

        let mut kb = 0.0;
        for i in 0..n {
            let w_row: Vec<f64> = (0..n).map(|j| state.w_b[i * n + j]).collect();
            let u_row: Vec<f64> = (0..n).map(|j| prior_full[i * n + j]).collect();
            kb += kl_dirichlet(&w_row, &u_row);
        }
        kb
    };

    let ln_zs = ln_zq + ln_zq2 + ln_zz;
    let f = ln_zs - kl_pi - kl_diff - kl_b;

    (f, kl_pi, kl_diff, kl_b, ln_zs)
}

// =========================================================================
// Per-trajectory KL (isCalcKlEach=True)
// =========================================================================

fn compute_per_trajectory_kl(
    dx2: &[f64],
    state: &State,
    trj_starts: &[usize],
    trj_ends: &[usize],
    base: &Priors,
    dim: usize,
    mag: f64,
) -> (f64, f64, f64, f64) {
    use rayon::prelude::*;

    let n = state.n.len();
    let m = trj_starts.len();
    let t_total = dx2.len();

    // Reconstruct ln_h and ln_q from converged state
    let mut ln_h = vec![0.0; t_total * n];
    let mut ln_h0 = vec![0.0; n];
    let mut gamma_rate = vec![0.0; n];
    for j in 0..n {
        ln_h0[j] = (dim as f64 / 2.0) * (digamma(state.n[j]) - (std::f64::consts::PI * state.c[j]).ln());
        gamma_rate[j] = state.n[j] / state.c[j];
    }
    for t in 0..t_total {
        for j in 0..n {
            ln_h[t * n + j] = ln_h0[j] - gamma_rate[j] * dx2[t];
        }
    }
    let wpi_sum: f64 = state.w_pi.iter().sum();
    let ln_pi: Vec<f64> = state.w_pi.iter().map(|&w| digamma(w) - digamma(wpi_sum)).collect();
    for mi in 0..m {
        let s = trj_starts[mi];
        for j in 0..n {
            ln_h[s * n + j] += ln_pi[j];
        }
    }

    let mut ln_q = vec![0.0; n * n];
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| state.w_b[i * n + j]).sum();
        let dg_sum = digamma(row_sum);
        for j in 0..n {
            ln_q[i * n + j] = digamma(state.w_b[i * n + j]) - dg_sum;
        }
    }

    // Normalize Q and H
    let ln_q_max = ln_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let q_norm: Vec<f64> = ln_q.iter().map(|&v| (v - ln_q_max).exp()).collect();
    let mut ln_h_max = vec![f64::NEG_INFINITY; t_total];
    for t in 0..t_total {
        for j in 0..n { if ln_h[t * n + j] > ln_h_max[t] { ln_h_max[t] = ln_h[t * n + j]; } }
    }
    let mut h_norm = vec![0.0; t_total * n];
    for t in 0..t_total {
        for j in 0..n { h_norm[t * n + j] = (ln_h[t * n + j] - ln_h_max[t]).exp(); }
    }

    // Base w_b with diagonal
    let mut base_wb_full = base.w_b.clone();
    if n > 1 {
        let off_diag = base.w_b[1]; // [0,1]
        for i in 0..n { base_wb_full[i * n + i] = off_diag * mag; }
    }

    // Parallel per-trajectory KL
    let results: Vec<(f64, f64, f64)> = (0..m)
        .into_par_iter()
        .map(|mi| {
            let s = trj_starts[mi];
            let e = trj_ends[mi];
            let tl = e - s + 1;

            let h_slice: Vec<f64> = h_norm[s * n..(e + 1) * n].to_vec();
            let (_lnz, wa_m, pst_m) = fb_single(&q_norm, &h_slice, tl, n);

            // Per-trajectory emission posterior
            let mut n_m = base.n.clone();
            let mut c_m = base.c.clone();
            for j in 0..n {
                let mut pst_sum = 0.0;
                let mut pst_dx2 = 0.0;
                for t in 0..tl {
                    pst_sum += pst_m[t * n + j];
                    pst_dx2 += pst_m[t * n + j] * dx2[s + t];
                }
                n_m[j] += (dim as f64 / 2.0) * pst_sum;
                c_m[j] += pst_dx2;
            }

            // Per-trajectory initial state posterior
            let w_pi_m: Vec<f64> = (0..n).map(|j| base.w_pi[j] + pst_m[j]).collect();

            // Per-trajectory transition posterior
            let w_b_m: Vec<f64> = (0..n * n).map(|k| base_wb_full[k] + wa_m[k]).collect();

            // KL computations
            let kl_pi_m = kl_dirichlet(&w_pi_m, &base.w_pi);
            let kl_diff_m = kl_diffusion(&n_m, &c_m, &base.n, &base.c);
            let kl_b_m = if n <= 1 {
                0.0
            } else {
                let mut kb = 0.0;
                for i in 0..n {
                    let w_row: Vec<f64> = (0..n).map(|j| w_b_m[i * n + j]).collect();
                    let u_row: Vec<f64> = (0..n).map(|j| base_wb_full[i * n + j]).collect();
                    kb += kl_dirichlet(&w_row, &u_row);
                }
                kb
            };
            (kl_pi_m, kl_diff_m, kl_b_m)
        })
        .collect();

    let mut kl_pi_total = 0.0;
    let mut kl_diff_total = 0.0;
    let mut kl_b_total = 0.0;
    for (kp, kd, kb) in &results {
        kl_pi_total += kp;
        kl_diff_total += kd;
        kl_b_total += kb;
    }
    let kl_total = kl_pi_total + kl_diff_total + kl_b_total;
    (kl_total, kl_pi_total, kl_diff_total, kl_b_total)
}

// =========================================================================
// Main VBEM loop (single N, single run)
// =========================================================================

pub fn run_single(
    dx2: &[f64],
    trj_starts: &[usize],
    trj_ends: &[usize],
    n_states: usize,
    n_traj: usize,
    params: &VBHMMParams,
    _seed: u64,
    // K-means++ is fully deterministic in this implementation:
    //   - first centroid: median of input data
    //   - subsequent centroids: farthest-point heuristic
    // No randomness is used, so seed has no effect. This parameter is
    // kept only for signature compatibility; remove when safe.
) -> VBHMMSingleResult {
    let priors = build_priors(n_states, n_traj, params, true);
    let base = build_priors(n_states, 1, params, false);
    // The REPORTED bound `f` measures its KL against `base` (unscaled) while
    // the M-step fits with `priors` (M-scaled), so it is the bound of a
    // different model than the one being fitted and the VBEM monotonicity
    // theorem does not cover it: measured, it rises, peaks, then descends to
    // the fixed point.  Convergence is judged on `f_conv` instead, which
    // evaluates lnZ and the KL at the same state and against `priors`.  That
    // combination is monotone to machine epsilon.  Both corrections are
    // needed; either alone leaves the sequence non-monotone.
    // See smda/core/vbhmm.py:run_vbhmm and scripts/b1_monotonicity_matrix.py.
    //
    // `f` itself is unchanged: it feeds model selection and hmm.csv, and it is
    // what reproduces AAS's BestN.
    // Per-trajectory KL (is_calc_kl_each) is computed post-convergence.
    let mut state = kmeans_init(dx2, trj_starts, trj_ends, n_states, &priors, params);

    let mut f_old = f64::NEG_INFINITY;
    let mut f_conv_old = f64::NEG_INFINITY;
    let mut converged = false;
    let mut pst_out = vec![0.0; dx2.len() * n_states];
    let mut kl_pi = 0.0;
    let mut kl_diff = 0.0;
    let mut kl_b = 0.0;
    let mut ln_zs = 0.0;
    let mut n_iter = 0;

    for iter in 0..params.max_iter {
        n_iter = iter + 1;
        let old_n = state.n.clone();
        let old_c = state.c.clone();

        // E-step
        let (pst, wa, lnzz, lnzq, lnzq2) = e_step(dx2, &state, trj_starts, trj_ends, SPATIAL_DIM);

        // Convergence bound: lnZ and KL both at the state the E-step just ran
        // on, KL against the same scaled priors the M-step uses.
        let (f_conv, _, _, _, _) =
            compute_lower_bound(lnzq, lnzq2, lnzz, &state, &priors, params.mag);

        // M-step
        state = m_step(dx2, &pst, &wa, &priors, trj_starts, SPATIAL_DIM);

        // Reported bound (unscaled base priors; feeds model selection)
        let (f, kp, kd, kb, lz) = compute_lower_bound(lnzq, lnzq2, lnzz, &state, &base, params.mag);
        kl_pi = kp; kl_diff = kd; kl_b = kb; ln_zs = lz;
        pst_out = pst;

        // Convergence check
        if f_conv_old != f64::NEG_INFINITY {
            let rel_change = (f_conv - f_conv_old).abs() / f_conv.abs().max(1e-300);
            let mut param_change = 0.0f64;
            for j in 0..n_states {
                let r_n = (state.n[j] - old_n[j]).abs() / old_n[j].abs().max(1e-300);
                let r_c = (state.c[j] - old_c[j]).abs() / old_c[j].abs().max(1e-300);
                param_change = param_change.max(r_n).max(r_c);
            }
            if rel_change < VBEM_CONV_TOL && param_change < 1e-2 {
                converged = true;
                f_old = f;
                break;
            }
        }
        f_conv_old = f_conv;
        f_old = f;
    }

    // Per-trajectory KL (isCalcKlEach=True): recompute KL post-convergence
    if params.is_calc_kl_each {
        let (kt, kp, kd, kb) = compute_per_trajectory_kl(
            dx2, &state, trj_starts, trj_ends, &base, SPATIAL_DIM, params.mag,
        );
        kl_pi = kp; kl_diff = kd; kl_b = kb;
        f_old = ln_zs - kt;
    }

    // D values
    // E[D] = c / [4·dt·(n-1)] — true posterior expectation under γ ~ Gamma(n, c)
    // (paper Eq. 16, matches AAS2 to floating-point precision)
    let d_values: Vec<f64> = (0..n_states)
        .map(|j| state.c[j] / (4.0 * params.timestep * (state.n[j] - 1.0)))
        .collect();

    VBHMMSingleResult {
        lower_bound: f_old,
        ln_zs, kl: kl_pi + kl_diff + kl_b,
        kl_pi, kl_diffusion: kl_diff, kl_b,
        state, pst: pst_out, d_values,
        converged, n_iter,
    }
}

// =========================================================================
// Model selection (minHidden..maxHidden)
// K-means++ initialization is fully deterministic, so a single run per
// n_states is sufficient — multiple restarts would produce identical results.
// =========================================================================

pub fn run_model_selection(
    dx2: &[f64],
    trj_starts: &[usize],
    trj_ends: &[usize],
    n_traj: usize,
    params: &VBHMMParams,
    min_hidden: usize,
    max_hidden: usize,
    seed: u64,
) -> VBHMMFullResult {
    let mut models = Vec::new();

    for n_states in min_hidden..=max_hidden {
        let result = run_single(dx2, trj_starts, trj_ends,
                                 n_states, n_traj, params, seed);
        models.push(result);
    }

    // Select best model by lower bound
    let mut best_idx = 0;
    let mut best_f = f64::NEG_INFINITY;
    for (i, m) in models.iter().enumerate() {
        if m.lower_bound.is_finite() && m.lower_bound > best_f {
            best_f = m.lower_bound;
            best_idx = i;
        }
    }
    let best_n = models[best_idx].d_values.len();

    VBHMMFullResult { models, best_n }
}

// =========================================================================
// State assignment (MaxProb)
// =========================================================================

pub fn assign_states_maxprob(
    pst: &[f64],     // T×N
    d_values: &[f64], // N
    trj_starts: &[usize],
    trj_ends: &[usize],
    n_states: usize,
) -> Vec<i32> {
    let t = pst.len() / n_states;

    // Sort states by D ascending → mapping old→new (1-indexed)
    let mut d_order: Vec<(f64, usize)> = d_values.iter().cloned().enumerate().map(|(i, d)| (d, i)).collect();
    d_order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut old_to_new = vec![0i32; n_states];
    for (new_k, &(_, old_k)) in d_order.iter().enumerate() {
        old_to_new[old_k] = (new_k + 1) as i32;
    }

    // Argmax per step
    let mut step_states = vec![0i32; t];
    for ti in 0..t {
        let mut best_j = 0;
        let mut best_v = f64::NEG_INFINITY;
        for j in 0..n_states {
            if pst[ti * n_states + j] > best_v {
                best_v = pst[ti * n_states + j];
                best_j = j;
            }
        }
        step_states[ti] = old_to_new[best_j];
    }

    // Mark last step of each trajectory as 0
    for &e in trj_ends {
        if e < t { step_states[e] = 0; }
    }

    step_states
}
