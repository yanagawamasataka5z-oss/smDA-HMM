"""Tests for VBHMM implementation.

Validates against AAS v2.1 SimplevbSPT output using Sample data.

Tests marked @pytest.mark.slow require Sample data and run full VBHMM analysis
(~10-15 minutes without Numba).

Note: run_vbhmm_analysis is deprecated. It is kept for reference/regression.

INITIALISATION CHANGE (2026-08-31)
----------------------------------
The Python path used stochastic K-means++ (rng.integers / rng.choice, Lloyd in
log space) until 2026-08-31. Rust had been deterministic since S41 (commit
2653179: median + farthest point, Lloyd on linear dx2); the Python copy was
never updated. That was a missed update, not a deliberate second
implementation — the Python path is NOT an independent reference.

Python now runs the identical deterministic algorithm, so both implementations
agree bit for bit (verified: all 9 validation cells match down to the iteration
count, D difference 0.0 ppm).

Effect on this file: the Python results shifted slightly — for data/Sample the
D error against AAS moved from 218.7 ppm to 136.8 ppm, i.e. onto the Rust
value. **No expected value in this file needed rewriting**: every assertion
here uses a tolerance (1-15%) far wider than that shift, and the exact-match
assertion (test_state_assignment_counts, state 0 count == 2029) is a
combinatorial property that does not depend on the optimum reached. Only
TestNumRun changed, because its premise (restarts explore different optima)
stopped being true; see the note on that class.
"""

import warnings
import numpy as np
import pytest
from pathlib import Path

# Suppress deprecation warning from run_vbhmm_analysis in all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:run_vbhmm_analysis.*deprecated:DeprecationWarning"
)

from smda_hmm.vbhmm.model import (
    VBHMMParams,
    VBHMMState,
    preprocess_trajectories,
    run_vbhmm,
    run_vbhmm_analysis,
    assign_states,
    _build_priors,
    _build_base_priors,
    vbem_step,
    forward_backward,
    SPATIAL_DIM,
)
from tests.helpers import make_test_vbhmm_params, v4_sample_dir

# --- AAS Model 4 reference values (from Sample_hmm.csv) ---
AAS_MODEL4 = {
    "D": np.array([0.005131619, 0.01048658, 0.03453339, 0.21981406]),
    "pi": np.array([0.21100219, 0.21976702, 0.2599213, 0.30930945]),
    "A": np.array([
        [0.9653565, 0.017133037, 0.00934134, 0.008169092],
        [0.01601743, 0.9077208, 0.0637555, 0.012506303],
        [0.004148171, 0.028464923, 0.8369158, 0.13047108],
        [0.003126123, 0.004847688, 0.12613863, 0.8658876],
    ]),
    "n": np.array([11763.727, 12356.541, 25575.973, 27117.998]),
    "c": np.array([8.040196, 17.258375, 117.640976, 793.96497]),
    "wPi": np.array([2140.6174, 2229.5366, 2636.9019, 3137.9446]),
    "wB": np.array([
        [9944.651, 176.49654, 96.23011, 84.15417],
        [173.46298, 9830.288, 690.449, 135.43875],
        [97.69835, 670.4102, 19711.168, 3072.8748],
        [77.10742, 119.57071, 3111.2737, 21357.56],
    ]),
    "F": 189169.10,
    "lnZs": 189273.34,
    "kl_pi": 10.5842285,
    "kl_diff": 21.185547,
    "kl_b": 72.48633,
}

AAS_LB_ALL = {1: 148280.06, 2: 182651.27, 3: 188421.95, 4: 189169.10, 5: 189087.52}

# The reference numbers above were measured by AAS on smDA-Python's Sample,
# which is v4 and is not part of this deposit.  Re-anchoring them to a bundled
# cell would mean replacing every expected value with one this implementation
# produced, which tests nothing.  So the data is pointed at instead; see
# tests.helpers.v4_sample_dir.  The path used to be a fixed data/Sample, which
# does not exist here, and these 29 tests skipped silently from the port until
# 2026-09-02.
_V4_DIR = v4_sample_dir()
SAMPLE_DATA = _V4_DIR / "Sample_data.csv" if _V4_DIR else None
SAMPLE_HMM = _V4_DIR / "Sample_hmm.csv" if _V4_DIR else None

SKIP_NO_SAMPLE = pytest.mark.skipif(
    _V4_DIR is None,
    reason="set SMDA_V4_SAMPLE to a directory holding an AAS v4 Sample pair"
)


@pytest.fixture(scope="module")
def default_params():
    return make_test_vbhmm_params()


@pytest.fixture(scope="module")
def sample_data(default_params):
    return preprocess_trajectories(SAMPLE_DATA, default_params)


# ---------------------------------------------------------------------------
# Sentinel tests (Tier 1.3)
# ---------------------------------------------------------------------------


class TestVBHMMParamsSentinel:

    def test_validate_detects_sentinel(self):
        """sentinel が残っていれば ValueError"""
        p = VBHMMParams()
        with pytest.raises(ValueError, match="VBHMMParams fields not set"):
            p.validate()

    def test_validate_passes_with_valid(self):
        """Category A フィールドが有効値なら pass"""
        p = make_test_vbhmm_params()
        p.validate()

    def test_default_construction_raises_in_run(self):
        """run_vbhmm_analysis に params=None を渡すと ValueError"""
        with pytest.raises(ValueError, match="VBHMMParams must be provided"):
            run_vbhmm_analysis("dummy.csv", params=None)


# ---------------------------------------------------------------------------
# Test preprocessing
# ---------------------------------------------------------------------------

@SKIP_NO_SAMPLE
class TestPreprocess:
    def test_trajectory_count(self, sample_data):
        """Sample data has 2029 trajectories."""
        assert sample_data.n_trajectories == 2029

    def test_step_count(self, sample_data):
        """Total steps = total_frames - n_trajectories = 70729 - 2029 = 68700."""
        assert sample_data.n_steps == 70729 - 2029

    def test_dx2_positive(self, sample_data):
        """All dx² values should be non-negative."""
        assert np.all(sample_data.dx2 >= 0)

    def test_trajectory_boundaries(self, sample_data):
        """trj_starts and trj_ends should be consistent."""
        assert len(sample_data.trj_starts) == sample_data.n_trajectories
        assert len(sample_data.trj_ends) == sample_data.n_trajectories
        # First start should be 0
        assert sample_data.trj_starts[0] == 0
        # Last end should be n_steps - 1
        assert sample_data.trj_ends[-1] == sample_data.n_steps - 1
        # Each start should be after previous end
        for i in range(1, sample_data.n_trajectories):
            assert sample_data.trj_starts[i] == sample_data.trj_ends[i - 1] + 1


# ---------------------------------------------------------------------------
# Test single model (Model 1)
# ---------------------------------------------------------------------------

@SKIP_NO_SAMPLE
class TestModel1:
    @pytest.fixture(scope="class")
    def model1(self, sample_data, default_params):
        return run_vbhmm(sample_data, 1, default_params)

    def test_lower_bound(self, model1):
        """Model 1 F ≈ 148280.06 ± 10."""
        assert abs(model1.lower_bound - 148280.06) < 50, (
            f"Model 1 F = {model1.lower_bound:.2f}, expected ~148280.06"
        )

    def test_converged(self, model1):
        assert model1.converged

    def test_single_state_d(self, model1):
        """D ≈ 0.09880 μm²/s (from AAS output)."""
        assert abs(model1.D[0] - 0.09880) < 0.005, (
            f"D = {model1.D[0]:.6f}, expected ~0.09880"
        )


# ---------------------------------------------------------------------------
# Test VBEM from AAS-initialized parameters (algorithm validation)
# ---------------------------------------------------------------------------

@SKIP_NO_SAMPLE
class TestVBEMFromAAS:
    """Verify VBEM step from AAS-converged parameters.

    Eliminates K-means randomness — validates the core VBEM algorithm.
    """

    @pytest.fixture(scope="class")
    def vbem_result(self, sample_data, default_params):
        """Run one VBEM step from AAS Model 4 converged parameters."""
        state_aas = VBHMMState(
            n=AAS_MODEL4["n"].copy(),
            c=AAS_MODEL4["c"].copy(),
            w_pi=AAS_MODEL4["wPi"].copy(),
            w_b=AAS_MODEL4["wB"].copy(),
        )
        priors = _build_priors(4, sample_data.n_trajectories, default_params)
        base_priors = _build_base_priors(4, default_params)
        new_state, pst, F, kl_pi, kl_diff, kl_b, ln_zs = vbem_step(
            sample_data.dx2, state_aas, priors, base_priors,
            sample_data.trj_starts, sample_data.trj_ends, SPATIAL_DIM)
        return {
            "state": new_state, "pst": pst,
            "F": F, "kl_pi": kl_pi, "kl_diff": kl_diff, "kl_b": kl_b,
            "ln_zs": ln_zs,
        }

    # --- Lower Bound ---
    def test_lower_bound(self, vbem_result):
        assert abs(vbem_result["F"] - AAS_MODEL4["F"]) < 10, (
            f"F = {vbem_result['F']:.2f}, expected ~{AAS_MODEL4['F']:.2f}"
        )

    def test_ln_zs(self, vbem_result):
        assert abs(vbem_result["ln_zs"] - AAS_MODEL4["lnZs"]) < 10, (
            f"lnZs = {vbem_result['ln_zs']:.2f}, expected ~{AAS_MODEL4['lnZs']}"
        )

    # --- KL components ---
    def test_kl_pi(self, vbem_result):
        assert abs(vbem_result["kl_pi"] - AAS_MODEL4["kl_pi"]) < 2.0, (
            f"kl_pi = {vbem_result['kl_pi']:.4f}, expected ~{AAS_MODEL4['kl_pi']}"
        )

    def test_kl_diffusion(self, vbem_result):
        assert abs(vbem_result["kl_diff"] - AAS_MODEL4["kl_diff"]) < 2.0, (
            f"kl_diff = {vbem_result['kl_diff']:.4f}, expected ~{AAS_MODEL4['kl_diff']}"
        )

    def test_kl_b(self, vbem_result):
        assert abs(vbem_result["kl_b"] - AAS_MODEL4["kl_b"]) < 10.0, (
            f"kl_b = {vbem_result['kl_b']:.4f}, expected ~{AAS_MODEL4['kl_b']}"
        )

    # --- Diffusion coefficients ---
    def test_d_values(self, vbem_result, default_params):
        state = vbem_result["state"]
        D = np.sort(state.c / (4 * default_params.timestep * state.n))
        for i, (d, exp) in enumerate(zip(D, AAS_MODEL4["D"])):
            assert abs(d - exp) / exp < 0.01, (
                f"D[{i}] = {d:.6f}, expected ~{exp:.6f}"
            )

    # --- Variational parameters (n, c, wPi, wB) ---
    def test_variational_n(self, vbem_result):
        n_sorted = np.sort(vbem_result["state"].n)
        n_expected = np.sort(AAS_MODEL4["n"])
        np.testing.assert_allclose(n_sorted, n_expected, rtol=0.05,
                                   err_msg="Variational n mismatch")

    def test_variational_c(self, vbem_result):
        c_sorted = np.sort(vbem_result["state"].c)
        c_expected = np.sort(AAS_MODEL4["c"])
        np.testing.assert_allclose(c_sorted, c_expected, rtol=0.05,
                                   err_msg="Variational c mismatch")

    def test_variational_wpi(self, vbem_result):
        wpi = vbem_result["state"].w_pi
        # Sort by wPi ascending to match ordering
        wpi_sorted = np.sort(wpi)
        # Expected values from smda with P5 scaling (w_pi_scale_mode=1, w_b_scale_mode=1).
        # Session 42 Phase 2.A'-4: adopted P5 pattern based on AAS reverse analysis.
        # These values now closely match AAS hmm.csv (diff < 0.003).
        wpi_expected = np.array([2140.61803914, 2229.53765022, 2636.89942384, 3137.94488680])
        np.testing.assert_allclose(wpi_sorted, wpi_expected, rtol=0.05,
                                   err_msg="Variational wPi mismatch")

    def test_variational_wb(self, vbem_result, default_params):
        """wB diagonal (stay) values within 10% after single VBEM step."""
        state = vbem_result["state"]
        D = state.c / (4 * default_params.timestep * state.n)
        d_order = np.argsort(D)
        wB_sorted = state.w_b[np.ix_(d_order, d_order)]
        for i in range(4):
            assert abs(wB_sorted[i, i] - AAS_MODEL4["wB"][i, i]) / AAS_MODEL4["wB"][i, i] < 0.10, (
                f"wB[{i},{i}] = {wB_sorted[i,i]:.1f}, expected ~{AAS_MODEL4['wB'][i,i]:.1f}"
            )

    # --- Transition matrix A ---
    def test_transition_matrix(self, vbem_result, default_params):
        """Transition matrix A = wB / rowsum(wB), ±0.03 for diagonal, ±0.02 off-diag."""
        state = vbem_result["state"]
        D = state.c / (4 * default_params.timestep * state.n)
        d_order = np.argsort(D)
        wB_sorted = state.w_b[np.ix_(d_order, d_order)]
        A = wB_sorted / wB_sorted.sum(axis=1, keepdims=True)
        A_expected = AAS_MODEL4["A"]

        max_diag_err = 0.0
        max_offdiag_err = 0.0
        for i in range(4):
            for j in range(4):
                err = abs(A[i, j] - A_expected[i, j])
                if i == j:
                    max_diag_err = max(max_diag_err, err)
                    assert err < 0.03, (
                        f"A[{i},{j}] = {A[i,j]:.4f}, expected {A_expected[i,j]:.4f}, err={err:.4f}"
                    )
                else:
                    max_offdiag_err = max(max_offdiag_err, err)
                    assert err < 0.02, (
                        f"A[{i},{j}] = {A[i,j]:.4f}, expected {A_expected[i,j]:.4f}, err={err:.4f}"
                    )

    # --- Initial probability π ---
    def test_initial_probability(self, vbem_result):
        wpi = vbem_result["state"].w_pi
        pi = wpi / np.sum(wpi)
        pi_sorted = np.sort(pi)
        # Expected from smda with P5 scaling (w_pi_scale_mode=1, w_b_scale_mode=1).
        # Now closely matches AAS π (~0.21-0.31).
        pi_expected = np.array([0.211002, 0.219767, 0.259921, 0.309310])
        np.testing.assert_allclose(pi_sorted, pi_expected, atol=0.05,
                                   err_msg="Initial probability mismatch")

    # --- Dwell times τ_dwell = 1 / (1 - A[j,j]) ---
    def test_dwell_times(self, vbem_result, default_params):
        """τ_dwell in frames: S1~28.9, S2~10.8, S3~6.1, S4~7.5 (±20%)."""
        state = vbem_result["state"]
        D = state.c / (4 * default_params.timestep * state.n)
        d_order = np.argsort(D)
        wB_sorted = state.w_b[np.ix_(d_order, d_order)]
        A = wB_sorted / wB_sorted.sum(axis=1, keepdims=True)

        tau_dwell = 1.0 / (1.0 - np.diag(A))  # in frames
        expected_tau = np.array([28.9, 10.8, 6.1, 7.5])
        for i, (tau, exp) in enumerate(zip(tau_dwell, expected_tau)):
            assert abs(tau - exp) / exp < 0.20, (
                f"τ_dwell[{i}] = {tau:.1f} frames, expected ~{exp:.1f}"
            )


# ---------------------------------------------------------------------------
# Test full analysis
# ---------------------------------------------------------------------------

@SKIP_NO_SAMPLE
class TestFullAnalysis:
    @pytest.fixture(scope="class")
    def result(self):
        return run_vbhmm_analysis(SAMPLE_DATA, params=make_test_vbhmm_params(),
                                  num_run=1)

    def test_suitable_model(self, result):
        """Best model should be N=4 or N=5.

        K-means initialization randomness can cause Model 5 to slightly
        beat Model 4. Both are acceptable (the data has 4 true states).
        """
        assert result.best_model in (4, 5), (
            f"Best model = {result.best_model}, expected 4 or 5"
        )

    def test_model1_lower_bound(self, result):
        """Model 1 F should match AAS exactly (no K-means dependence)."""
        model1 = result.models[0]
        assert abs(model1.lower_bound - 148280.06) < 10, (
            f"Model 1 F = {model1.lower_bound:.2f}, expected ~148280.06"
        )

    def test_all_lower_bounds_ordering(self, result):
        """F should increase with N up to the best model."""
        bounds = [m.lower_bound for m in result.models]
        # Models 1-4 should have increasing F
        for i in range(min(3, len(bounds) - 1)):
            assert bounds[i + 1] > bounds[i], (
                f"F({i+2}) = {bounds[i+1]:.2f} <= F({i+1}) = {bounds[i]:.2f}"
            )

    def test_model4_diffusion(self, result):
        """Model 4 D values should match AAS within 15%."""
        model4 = result.models[3]  # 0-indexed
        D_sorted = np.sort(model4.D)
        expected_D = [0.005132, 0.010487, 0.034533, 0.219814]
        for i, (d, exp) in enumerate(zip(D_sorted, expected_D)):
            assert abs(d - exp) / exp < 0.15, (
                f"D[{i}] = {d:.6f}, expected ~{exp:.6f}"
            )

    def test_model4_transition(self, result):
        """Model 4 A[1,1] (slowest state self-transition) ≈ 0.9654."""
        model4 = result.models[3]
        d_order = np.argsort(model4.D)
        wB_sorted = model4.state.w_b[np.ix_(d_order, d_order)]
        A11 = wB_sorted[0, 0] / np.sum(wB_sorted[0])
        assert abs(A11 - 0.9654) < 0.02, f"A[1,1] = {A11:.4f}, expected ~0.9654"

    def test_state_assignment_counts(self, result):
        """Check state assignment distribution for 4-state model."""
        states = result.state_assignments[4]
        # State 0 count should be trajectory count
        n_zero = np.sum(states == 0)
        assert n_zero == 2029, f"State 0 count = {n_zero}, expected 2029"

        # Total assigned (non-zero) should be total_frames - n_trajectories
        n_assigned = np.sum(states > 0)
        assert n_assigned == 70729 - 2029


# ---------------------------------------------------------------------------
# Test synthetic data (small scale, no file dependency)
# ---------------------------------------------------------------------------

class TestSynthetic:
    def test_forward_backward_normalization(self):
        """Posterior probabilities should sum to 1 at each time step."""
        rng = np.random.default_rng(123)
        T, N = 20, 2
        ln_h = rng.standard_normal((T, N))
        ln_q = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
        trj_starts = np.array([0])
        trj_ends = np.array([T - 1])

        pst, wA, lnZz = forward_backward(ln_h, ln_q, trj_starts, trj_ends)

        # Check normalization
        row_sums = np.sum(pst, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

        # wA should be non-negative
        assert np.all(wA >= 0)

    def test_transition_matrix_row_sum(self):
        """Transition matrix rows must sum to 1.0 after normalization."""
        rng = np.random.default_rng(42)
        # Create synthetic wB with known structure
        wB = np.array([
            [100.0, 5.0, 2.0],
            [3.0, 80.0, 10.0],
            [1.0, 8.0, 60.0],
        ])
        A = wB / wB.sum(axis=1, keepdims=True)
        row_sums = A.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_lower_bound_monotonic(self):
        """Lower bound should generally increase (or stay same) over VBEM iterations."""
        from smda_hmm.vbhmm.model import (
            _build_priors, _build_base_priors, kmeans_init, vbem_step,
            VBHMMParams, TrajectoryData,
        )

        rng = np.random.default_rng(42)
        params = make_test_vbhmm_params(max_iter=20)

        # Generate simple 2-state synthetic data
        n_trj = 50
        steps_per_trj = 20
        dx2_list = []
        trj_starts = []
        trj_ends = []
        offset = 0
        for _ in range(n_trj):
            # Randomly assign trajectory to slow or fast
            D = 0.01 if rng.random() < 0.5 else 0.1
            dt = 0.0333
            sigma = np.sqrt(2 * D * dt)
            dx = rng.normal(0, sigma, steps_per_trj)
            dy = rng.normal(0, sigma, steps_per_trj)
            dx2 = dx**2 + dy**2
            dx2_list.append(dx2)
            trj_starts.append(offset)
            trj_ends.append(offset + steps_per_trj - 1)
            offset += steps_per_trj

        dx2_all = np.concatenate(dx2_list)
        trj_starts_arr = np.array(trj_starts)
        trj_ends_arr = np.array(trj_ends)

        data = TrajectoryData(
            dx2=dx2_all,
            trj_ends=trj_ends_arr,
            trj_starts=trj_starts_arr,
            n_trajectories=n_trj,
            n_steps=len(dx2_all),
            roi=np.zeros(offset),
            x_px=np.zeros(offset),
            y_px=np.zeros(offset),
            frame=np.zeros(offset),
        )

        n_states = 2
        priors = _build_priors(n_states, n_trj, params)
        base_priors = _build_base_priors(n_states, params)
        # rng was removed on 2026-08-31: the Python initialisation is now the
        # same deterministic algorithm as Rust (median + farthest point,
        # Lloyd on linear dx2). It previously used stochastic k-means++.
        state = kmeans_init(data, n_states, priors, params)

        bounds = []
        for _ in range(10):
            state, pst, F, *_ = vbem_step(
                dx2_all, state, priors, base_priors,
                trj_starts_arr, trj_ends_arr, SPATIAL_DIM)
            bounds.append(F)

        # Check that F is generally non-decreasing (allow small numerical noise)
        for i in range(1, len(bounds)):
            assert bounds[i] >= bounds[i - 1] - 1.0, (
                f"F decreased: {bounds[i]:.2f} < {bounds[i-1]:.2f} at step {i}"
            )


# ---------------------------------------------------------------------------
# Test Model 5 NaN fix
# ---------------------------------------------------------------------------

class TestModel5Stability:
    def test_model5_no_nan(self):
        """5-state model Lower Bound should not be NaN even with limited data."""
        from smda_hmm.vbhmm.model import (
            _build_priors, _build_base_priors, kmeans_init, vbem_step,
            VBHMMParams, TrajectoryData,
        )

        rng = np.random.default_rng(42)
        params = make_test_vbhmm_params()

        # Generate synthetic 3-state data (5-state model will have empty states)
        n_trj = 30
        steps_per_trj = 8
        D_true = [0.01, 0.05, 0.2]

        dx2_list = []
        trj_starts = []
        trj_ends = []
        offset = 0
        for _ in range(n_trj):
            trj_starts.append(offset)
            d = rng.choice(D_true)
            dx2 = rng.exponential(scale=4 * d * params.timestep, size=steps_per_trj)
            dx2_list.append(dx2)
            trj_ends.append(offset + steps_per_trj - 1)
            offset += steps_per_trj

        data = TrajectoryData(
            dx2=np.concatenate(dx2_list),
            trj_ends=np.array(trj_ends),
            trj_starts=np.array(trj_starts),
            n_trajectories=n_trj,
            n_steps=offset,
            roi=np.zeros(offset + n_trj),
            x_px=np.zeros(offset + n_trj),
            y_px=np.zeros(offset + n_trj),
            frame=np.zeros(offset + n_trj),
        )

        n_states = 5
        priors = _build_priors(n_states, n_trj, params)
        base_priors = _build_base_priors(n_states, params)
        # rng was removed on 2026-08-31: the Python initialisation is now the
        # same deterministic algorithm as Rust (median + farthest point,
        # Lloyd on linear dx2). It previously used stochastic k-means++.
        state = kmeans_init(data, n_states, priors, params)

        for _ in range(params.max_iter):
            state, pst, F, *_ = vbem_step(
                data.dx2, state, priors, base_priors,
                data.trj_starts, data.trj_ends, SPATIAL_DIM)

        assert np.isfinite(F), f"Model 5 F is not finite: {F}"


# ---------------------------------------------------------------------------
# Test num_run parameter
# ---------------------------------------------------------------------------

class TestNumRun:
    """num_run / seed no longer affect the result.

    WHY THIS TEST CHANGED (2026-08-31)
    ----------------------------------
    It used to assert that num_run=5 gives a lower bound >= num_run=1, which
    is the correct expectation for *stochastic* restarts: more random starts
    can only find an equal or better optimum.

    The Python initialisation was stochastic k-means++ until 2026-08-31, so
    restarts genuinely explored different optima. Rust had been deterministic
    since S41 (commit 2653179); the Python copy was simply never updated — a
    missed update, not a second opinion. Aligning Python to Rust removed the
    randomness, so restarts now all start from the same point and return the
    same model. num_run and seed became no-ops.

    The assertion was therefore replaced rather than merely relaxed: "more
    restarts is at least as good" is no longer a meaningful claim, whereas
    "restarts change nothing" is exactly what the alignment guarantees.
    """

    @SKIP_NO_SAMPLE
    def test_num_run_has_no_effect(self):
        """Restarts are pointless once the initialisation is deterministic."""
        _p = make_test_vbhmm_params()
        result_1 = run_vbhmm_analysis(SAMPLE_DATA, params=_p, num_run=1, seed=42)
        result_5 = run_vbhmm_analysis(SAMPLE_DATA, params=_p, num_run=5, seed=42)

        for m1, m5 in zip(result_1.models, result_5.models):
            assert m1.lower_bound == m5.lower_bound, (
                f"N={m1.n_states}: num_run changed the lower bound "
                f"({m1.lower_bound} -> {m5.lower_bound}); the initialisation "
                f"is no longer deterministic."
            )
            np.testing.assert_array_equal(m1.D, m5.D)

    @SKIP_NO_SAMPLE
    def test_seed_has_no_effect(self):
        """Guards against reintroducing an rng-dependent initialisation."""
        _p = make_test_vbhmm_params()
        a = run_vbhmm_analysis(SAMPLE_DATA, params=_p, num_run=1, seed=42)
        b = run_vbhmm_analysis(SAMPLE_DATA, params=_p, num_run=1, seed=12345)
        for ma, mb in zip(a.models, b.models):
            assert ma.lower_bound == mb.lower_bound
            np.testing.assert_array_equal(ma.D, mb.D)

    @SKIP_NO_SAMPLE
    def test_num_run_1_backward_compatible(self):
        """num_run=1 with seed=42 should give same results as old behavior."""
        result = run_vbhmm_analysis(SAMPLE_DATA, params=make_test_vbhmm_params(),
                                    num_run=1, seed=42)

        # Model 1 F should match AAS exactly
        m1 = result.models[0]
        np.testing.assert_allclose(m1.lower_bound, AAS_LB_ALL[1], rtol=1e-4)
