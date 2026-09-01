"""Running the VB-HMM: parameter assembly, the Rust call, and result conversion.

These functions are the production compute path.  In smDA-Python they sat
inside the 29,000-line GUI module (`_run_vbhmm_rust`, `_convert_vbhmm_result`,
`_build_vbhmm_params`), which meant the algorithm could not be used without
importing Streamlit.  The pool worker and the parameter serialiser that went
with it are not carried over: smDA-HMM analyses files one at a time, so
pinning rayon to a single thread per process would only be slower.  Here they are plain functions: nothing in this module
touches `st.*` or session state.

The Rust engine (`smda_scan.run_vbhmm_model_selection`) is the production
implementation.  The pure-Python path in `model.py` computes the same thing and
is kept as a readable reference: the two were verified to agree to 2.4e-08 ppm
with identical iteration counts.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from smda_hmm.vbhmm.model import (
    VBHMMModelResult, VBHMMParams, VBHMMPriors, VBHMMResult, VBHMMState,
    assign_states, preprocess_trajectories,
)


def build_params(p: dict, timestep: float, dist_per_px: float):
    """Build VBHMMParams from the UI's dict.

    Upstream name: _build_vbhmm_params (smda/gui/pages.py).
    """
    return VBHMMParams(
        n_tilde=p["n_tilde"], c_tilde=p["c_tilde"],
        w_pi_tilde=p["wpi_tilde"], w_b_tilde=p["wb_tilde"],
        mag=p["mag"], max_hidden=p["max_states"], min_hidden=p["min_states"],
        max_iter=p["max_iter"], num_run=1,  # deterministic K-means++; multiple runs are identical
        timestep=timestep,
        distance_per_pixel=dist_per_px,
        is_add_each_trajectory=p["add_per_traj"],
        is_calc_kl_each=p["calc_kl_each"],
    )



def _convert_result(raw_result: dict, csv_path: str, params):
    """Convert the Rust return dict into a VBHMMResult. No computation.

    Upstream name: _convert_vbhmm_result (smda/gui/pages.py).
    """
    data = preprocess_trajectories(csv_path, params)

    models = []
    for md in raw_result['models']:
        n_st = md['n_states']
        D = np.array(md['D'])
        pst = np.array(md['pst'])
        # Rust posterior hyperparameters (no dummies, no fallback)
        _required_keys = ("state_n", "state_c", "state_w_pi", "state_w_b")
        _missing = [k for k in _required_keys if k not in md]
        if _missing:
            raise ValueError(
                f"Rust VBHMM return dict missing required keys: {_missing}. "
                f"Rust build may be outdated. Run: "
                f"cd smda-scan/smda-scan && maturin develop --release"
            )

        state = VBHMMState(
            n=np.asarray(md["state_n"], dtype=np.float64),
            c=np.asarray(md["state_c"], dtype=np.float64),
            w_pi=np.asarray(md["state_w_pi"], dtype=np.float64),
            w_b=np.asarray(md["state_w_b"], dtype=np.float64).reshape(
                n_st, n_st),
        )
        priors = VBHMMPriors(
            n=np.ones(n_st), c=np.ones(n_st),
            w_pi=np.ones(n_st), w_b=np.eye(n_st))

        for _name, _arr, _shape in [
            ("state_n", state.n, (n_st,)),
            ("state_c", state.c, (n_st,)),
            ("state_w_pi", state.w_pi, (n_st,)),
            ("state_w_b", state.w_b, (n_st, n_st)),
        ]:
            if _arr.shape != _shape:
                raise ValueError(
                    f"{_name} shape mismatch: expected {_shape}, "
                    f"got {_arr.shape}")

        m = VBHMMModelResult(
            n_states=n_st, lower_bound=md['lower_bound'],
            ln_zs=md['ln_zs'], kl=md['kl'],
            kl_pi=md['kl_pi'], kl_diffusion=md['kl_diffusion'],
            kl_b=md['kl_b'], state=state, priors=priors,
            pst=pst, converged=md['converged'],
            n_iter=md['n_iter'], D=D)
        models.append(m)

    # State assignments — use Python assign_states for correct
    # D-ascending sorting + 1-indexed states (boundary marker = 0)
    state_assignments = {}
    for model in models:
        state_assignments[model.n_states] = assign_states(model, data)

    return VBHMMResult(
        models=models, best_model=raw_result['best_n'],
        params=params, state_assignments=state_assignments)


def run(csv_path: str, params, seed: int = 42):
    """Run the VB-HMM through the Rust engine, returning a VBHMMResult.

    Upstream name: _run_vbhmm_rust (smda/gui/pages.py).
    """
    import smda_scan

    data = preprocess_trajectories(csv_path, params)

    result = smda_scan.run_vbhmm_model_selection(
        data.dx2.astype(np.float64),
        data.trj_starts.astype(np.int64),
        data.trj_ends.astype(np.int64),
        n_tilde=params.n_tilde,
        c_tilde=params.c_tilde,
        w_pi_tilde=params.w_pi_tilde,
        w_b_tilde=params.w_b_tilde,
        mag=params.mag,
        max_iter=params.max_iter,
        num_run=1,  # deterministic K-means++; multiple runs are identical
        timestep=params.timestep,
        min_hidden=params.min_hidden,
        max_hidden=params.max_hidden,
        seed=seed,
        w_pi_scale_mode=params.w_pi_scale_mode,
        w_b_scale_mode=params.w_b_scale_mode,
        is_calc_kl_each=params.is_calc_kl_each,
        is_add_each_trajectory=params.is_add_each_trajectory,
    )

    # Convert Rust dict → Python VBHMMResult
    models = []
    for md in result['models']:
        n_st = md['n_states']
        D = np.array(md['D'])
        pst = np.array(md['pst'])
        # Rust posterior hyperparameters (no dummies, no fallback)
        _required_keys = ("state_n", "state_c", "state_w_pi", "state_w_b")
        _missing = [k for k in _required_keys if k not in md]
        if _missing:
            raise ValueError(
                f"Rust VBHMM return dict missing required keys: {_missing}. "
                f"Rust build may be outdated. Run: "
                f"cd smda-scan/smda-scan && maturin develop --release"
            )

        state = VBHMMState(
            n=np.asarray(md["state_n"], dtype=np.float64),
            c=np.asarray(md["state_c"], dtype=np.float64),
            w_pi=np.asarray(md["state_w_pi"], dtype=np.float64),
            w_b=np.asarray(md["state_w_b"], dtype=np.float64).reshape(
                n_st, n_st),
        )
        priors = VBHMMPriors(
            n=np.ones(n_st), c=np.ones(n_st),
            w_pi=np.ones(n_st), w_b=np.eye(n_st),
        )

        for _name, _arr, _shape in [
            ("state_n", state.n, (n_st,)),
            ("state_c", state.c, (n_st,)),
            ("state_w_pi", state.w_pi, (n_st,)),
            ("state_w_b", state.w_b, (n_st, n_st)),
        ]:
            if _arr.shape != _shape:
                raise ValueError(
                    f"{_name} shape mismatch: expected {_shape}, "
                    f"got {_arr.shape}")

        m = VBHMMModelResult(
            n_states=n_st,
            lower_bound=md['lower_bound'],
            ln_zs=md['ln_zs'],
            kl=md['kl'],
            kl_pi=md['kl_pi'],
            kl_diffusion=md['kl_diffusion'],
            kl_b=md['kl_b'],
            state=state,
            priors=priors,
            pst=pst,
            converged=md['converged'],
            n_iter=md['n_iter'],
            D=D,
        )
        models.append(m)

    # State assignments — use Python assign_states for correct
    # D-ascending sorting + 1-indexed states (boundary marker = 0)
    state_assignments = {}
    for model in models:
        state_assignments[model.n_states] = assign_states(model, data)

    return VBHMMResult(
        models=models,
        best_model=result['best_n'],
        params=params,
        state_assignments=state_assignments,
    )
