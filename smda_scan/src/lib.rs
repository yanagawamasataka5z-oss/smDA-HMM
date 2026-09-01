//! smda-scan (smDA-HMM): VB-HMM diffusion-state inference.
//!
//! One binding, `run_vbhmm_model_selection`.  The crate in smDA-Python carries
//! twelve more modules for spot detection and tracking; none of them is here,
//! and `vbhmm.rs` is self-contained (std + statrs), so nothing was left behind
//! by dropping them.
//!
//! `vbhmm.rs` is byte-identical to the file it was verified against upstream.

mod vbhmm;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (
    dx2, trj_starts, trj_ends,
    n_tilde = 1.0, c_tilde = 0.001,
    w_pi_tilde = 1.0, w_b_tilde = 0.01,
    mag = 30.0, max_iter = 100, num_run = 5,
    timestep = 0.0333, min_hidden = 1, max_hidden = 5,
    seed = 42,
    *,
    w_pi_scale_mode,
    w_b_scale_mode,
    is_calc_kl_each = None,
    is_add_each_trajectory = None,
))]
fn run_vbhmm_model_selection<'py>(
    py: Python<'py>,
    dx2: PyReadonlyArray1<'py, f64>,
    trj_starts: PyReadonlyArray1<'py, i64>,
    trj_ends: PyReadonlyArray1<'py, i64>,
    n_tilde: f64,
    c_tilde: f64,
    w_pi_tilde: f64,
    w_b_tilde: f64,
    mag: f64,
    max_iter: usize,
    num_run: usize,
    timestep: f64,
    min_hidden: usize,
    max_hidden: usize,
    seed: u64,
    w_pi_scale_mode: u8,
    w_b_scale_mode: u8,
    is_calc_kl_each: Option<bool>,
    is_add_each_trajectory: Option<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    // Reject unimplemented scale modes instead of silently falling through to
    // "unscaled".  S69 recorded that a mis-set w_b_scale_mode produced a
    // flipped BestN and 23% D error, so a silent fallback here is not benign.
    if w_pi_scale_mode > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "w_pi_scale_mode={} is not implemented (valid: 0 = unscaled,              1 = scaled by trajectory count)",
            w_pi_scale_mode)));
    }
    if w_b_scale_mode > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "w_b_scale_mode={} is not implemented (valid: 0 = unscaled,              1 = scaled, AAS P5). Note: S69 renumbered these — the old mode 2              is the current mode 1.",
            w_b_scale_mode)));
    }
    let dx2_arr = dx2.as_array();
    let starts_arr = trj_starts.as_array();
    let ends_arr = trj_ends.as_array();

    let dx2_vec: Vec<f64> = dx2_arr.iter().copied().collect();
    let starts_vec: Vec<usize> = starts_arr.iter().map(|&v| v as usize).collect();
    let ends_vec: Vec<usize> = ends_arr.iter().map(|&v| v as usize).collect();
    let n_traj = starts_vec.len();

    let params = vbhmm::VBHMMParams {
        n_tilde, c_tilde, w_pi_tilde, w_b_tilde,
        mag, max_iter, num_run, timestep,
        w_pi_scale_mode, w_b_scale_mode,
        is_calc_kl_each: is_calc_kl_each.unwrap_or(false),
        is_add_each_trajectory: is_add_each_trajectory.unwrap_or(true),
    };

    // Run model selection (release GIL)
    let result = py.allow_threads(|| {
        vbhmm::run_model_selection(
            &dx2_vec, &starts_vec, &ends_vec,
            n_traj, &params, min_hidden, max_hidden, seed,
        )
    });

    // Build Python dict result
    let out = PyDict::new_bound(py);
    out.set_item("best_n", result.best_n)?;

    let models_list = pyo3::types::PyList::empty_bound(py);
    for model in &result.models {
        let md = PyDict::new_bound(py);
        let n_states = model.d_values.len();
        md.set_item("n_states", n_states)?;
        md.set_item("lower_bound", model.lower_bound)?;
        md.set_item("ln_zs", model.ln_zs)?;
        md.set_item("kl", model.kl)?;
        md.set_item("kl_pi", model.kl_pi)?;
        md.set_item("kl_diffusion", model.kl_diffusion)?;
        md.set_item("kl_b", model.kl_b)?;
        md.set_item("converged", model.converged)?;
        md.set_item("n_iter", model.n_iter)?;

        // D values
        let d_arr = PyArray1::from_slice_bound(py, &model.d_values);
        md.set_item("D", d_arr)?;

        // State assignments (MaxProb)
        let states = vbhmm::assign_states_maxprob(
            &model.pst, &model.d_values,
            &starts_vec, &ends_vec, n_states,
        );
        let states_arr = PyArray1::from_vec_bound(py, states);
        md.set_item("state_assignments", states_arr)?;

        // pst (T × N)
        let t_total = model.pst.len() / n_states;
        let pst_arr = PyArray2::zeros_bound(py, [t_total, n_states], false);
        {
            let mut m = unsafe { pst_arr.as_array_mut() };
            for t in 0..t_total {
                for j in 0..n_states {
                    m[[t, j]] = model.pst[t * n_states + j];
                }
            }
        }
        md.set_item("pst", pst_arr)?;

        // Posterior hyperparameters (state.n, state.c, state.w_pi, state.w_b)
        let state_n_arr = PyArray1::from_slice_bound(py, &model.state.n);
        md.set_item("state_n", state_n_arr)?;
        let state_c_arr = PyArray1::from_slice_bound(py, &model.state.c);
        md.set_item("state_c", state_c_arr)?;
        let state_wpi_arr = PyArray1::from_slice_bound(py, &model.state.w_pi);
        md.set_item("state_w_pi", state_wpi_arr)?;
        let state_wb_arr = PyArray1::from_slice_bound(py, &model.state.w_b);
        md.set_item("state_w_b", state_wb_arr)?;

        models_list.append(md)?;
    }
    out.set_item("models", models_list)?;

    Ok(out)
}


#[pymodule]
fn smda_scan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_vbhmm_model_selection, m)?)?;
    Ok(())
}
