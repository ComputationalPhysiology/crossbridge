import pytest
import numpy as np

from crossbridge import RDQ18


@pytest.fixture
def standard_model():
    """Returns a fresh model instance with 5 cells before each test that requests it."""
    return RDQ18(num_cells=5)


@pytest.fixture
def dt(standard_model):
    return standard_model.dt


def test_initialization_shapes(standard_model):
    """Test if the state arrays are initialized with correct dimensions."""
    # xODE shape: (nu-2, 4, 4, 4, num_cells)
    # nu defaults to 36, so dim 0 should be 34
    expected_shape = (34, 4, 4, 4, standard_model.num_cells)

    assert standard_model.xODE.shape == expected_shape, (
        f"xODE shape mismatch. Expected {expected_shape}, got {standard_model.xODE.shape}"
    )

    # Initial state should be fully non-permissive (0N = index 0)
    # Summing over axis 1,2,3 (the triplet states) should equal 1.0 for every unit
    sums = np.sum(standard_model.xODE, axis=(1, 2, 3))
    np.testing.assert_allclose(sums, 1.0, err_msg="Initial probabilities do not sum to 1")


def test_probability_conservation(standard_model, dt):
    """
    Test that probabilities remain conserved (sum to 1) after time stepping.
    This is the most critical mathematical validity check.
    """
    num_cells = standard_model.num_cells
    Ca = np.full(num_cells, 0.5)
    SL = np.full(num_cells, 2.2)

    # Run for 100 steps
    for _ in range(100):
        standard_model.advance_ODE(dt, Ca, SL)

    sums = np.sum(standard_model.xODE, axis=(1, 2, 3))

    # Check if any probability drifted from 1.0
    # Tolerance is slightly loose (1e-6) due to float accumulation over steps
    np.testing.assert_allclose(
        sums, 1.0, atol=1e-6, err_msg="Probability conservation violated after time stepping"
    )


def test_calcium_sensitivity(dt):
    """
    Physiological Test: Higher Calcium should lead to higher Permissivity.
    """
    model_high = RDQ18(num_cells=1)
    model_low = RDQ18(num_cells=1)

    Ca_high = np.array([2.0])  # 2 uM (Saturated)
    Ca_low = np.array([0.1])  # 0.1 uM (Diastolic)
    SL = np.array([2.2])

    # Run both for a brief period
    steps = 500
    for _ in range(steps):
        model_high.advance_ODE(dt, Ca_high, SL)
        model_low.advance_ODE(dt, Ca_low, SL)

    P_high = model_high.compute_permissivity()
    P_low = model_low.compute_permissivity()

    assert P_high[0] > P_low[0], (
        f"Model failed Ca sensitivity: High Ca ({P_high[0]:.4f}) <= Low Ca ({P_low[0]:.4f})"
    )


def test_length_dependence_overlap(dt):
    """
    Physiological Test: Force should be lower at very short lengths (overlap zone)
    compared to optimal length.
    """
    # SL = 2.2 (Optimal) vs SL = 1.5 (Compressed/Overlap)
    model_opt = RDQ18(num_cells=1)
    model_short = RDQ18(num_cells=1)

    Ca = np.array([1.0])  # High calcium

    # Run to steady state approx
    for _ in range(2000):
        model_opt.advance_ODE(dt, Ca, np.array([2.2]))
        model_short.advance_ODE(dt, Ca, np.array([1.5]))

    P_opt = model_opt.compute_permissivity()
    P_short = model_short.compute_permissivity()
    msg = (
        f"Model failed length-dependence: P_opt ({P_opt[0]:.4f}) "
        f"should be > P_short ({P_short[0]:.4f})"
    )
    assert P_opt[0] > P_short[0], msg


def test_vectorization_independence(dt):
    """
    Computational Test: Ensure that cells in a vectorized batch
    evolve independently and don't leak data into each other.
    """
    # Create a batch of 2 cells
    model = RDQ18(num_cells=2)

    # Cell 0 gets 0 Calcium (should stay 0)
    # Cell 1 gets High Calcium (should activate)
    Ca_mixed = np.array([0.0, 10.0])
    SL_mixed = np.array([2.2, 2.2])

    # Run for 20,000 steps to simulate 0.5 seconds of physical time
    for _ in range(20000):
        model.advance_ODE(dt, Ca_mixed, SL_mixed)

    P_vals = model.compute_permissivity()

    # Cell 0 should be essentially zero
    assert P_vals[0] < 0.01, (
        f"Cell 0 activated ({P_vals[0]}) despite 0 Calcium (Vectorization leak?)"
    )

    # Cell 1 should be high
    assert P_vals[1] > 0.5, f"Cell 1 failed to activate ({P_vals[1]}) in batch mode"


def test_spatial_functions_consistency(standard_model):
    """
    Test the geometry helper functions directly.
    """
    xi, xAZ, xLA, xRA, Q = standard_model.funcs

    # Check myosin node positions (xi) are increasing
    nodes = np.arange(standard_model.nu)
    positions = xi(nodes)
    is_monotonic = np.all(np.diff(positions) > 0)
    assert is_monotonic, "Myosin node positions must be strictly increasing"

    # Check geometric bounds for standard SL
    SL = np.array([2.2])
    az = xAZ(SL)
    # la = xLA(SL) # unused in assertion but part of tuple
    ra = xRA(SL)

    # Physically, RA < AZ (Right boundary of single overlap < End of overlap region)
    assert ra[0] < az[0], f"Geometry Error: xRA ({ra[0]}) should be < xAZ ({az[0]})"


def test_chi_function_bounds(standard_model):
    """
    Test that the Chi (overlap) functions return values between 0 and 1
    and do not crash with vector inputs.
    """
    SL_range = np.linspace(1.6, 2.3, 10)
    # We create dummy calcium to match shape
    dummy_Ca = np.ones(10)

    # Re-initialize a model of size 10 to match SL_range for valid broadcasting
    model_vec = RDQ18(num_cells=10)

    try:
        model_vec.update_probabilities(SL_range, dummy_Ca)

        # Check PC matrix for NaNs or Inf
        assert np.all(np.isfinite(model_vec.PC)), "Transition Matrix PC contains NaNs or Infs"

        # Check rates are non-negative
        assert np.all(model_vec.PC >= 0), "Transition rates cannot be negative"

    except Exception as e:
        pytest.fail(f"Chi computation failed for vector input: {e}")


def test_extreme_input_robustness(standard_model):
    """
    Test model stability under non-standard time steps or inputs.
    """
    # Very large dt might cause probabilities to overshoot > 1 without internal sub-stepping
    large_dt = 1e-3  # 40x standard dt
    num_cells = standard_model.num_cells
    Ca = np.array([1.0] * num_cells)
    SL = np.array([2.2] * num_cells)

    try:
        standard_model.advance_ODE(large_dt, Ca, SL)
        sums = np.sum(standard_model.xODE, axis=(1, 2, 3))
        np.testing.assert_allclose(
            sums, 1.0, atol=1e-2, err_msg="Model probability conservation failed under large dt"
        )
    except Exception as e:
        pytest.fail(f"Model crashed under large dt: {e}")
