import os
# Limit GPU memory allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # Use 75% of GPU memory

import time
from pathlib import Path

import numpy as np
import pytest
import re

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from phastphase.retrieval_jax import retrieve
from phastphase.retrieval_jax.alternative_methods._gradient_flows import *
from phastphase.retrieval_jax.alternative_methods._initializations import *


SELECTED_TEST_CASES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SELECTED_DESCENT_METHODS = [0]
SELECTED_WIND_METHODS = [0]
SELECTED_MAX_ITERS = [100]
SELECTED_GRAD_TOLERANCES = [1e-8]
SELECTED_SHOULD_GUESS_WIND = [False]
SELECTED_FOURIER_OVERSAMPLES = [2]
SELECTED_CONVERGENCE_TOLERANCES = [5e-4]
SELECTED_MAX_RANDOM_RESTARTS = [100]

RANDOM_GENERATOR = np.random.default_rng(0)


# (method_name, method_function, is_active?)
INITIAL_GUESS_METHODS = [
    ("random", random_initialization, False),
    ("spectral", spectral_initialization, False),
    ("trunc_spectral", truncated_spectral_initialization, True),
    ("ortho", orthogonality_promoting_initialization, False),
]


# (method_name, method_function, is_active?)
FLOW_METHODS = [
    ("wirtinger_flow", wirtinger_flow, True),
    ("truncated_wirtinger_flow", truncated_wirtinger_flow, True),
    ("amplitude_flow", amplitude_flow, True),
    ("truncated_amplitude_flow", truncated_amplitude_flow, True),
]


# Filter the list to only include active methods
ACTIVE_INIT_METHODS = [m for m in INITIAL_GUESS_METHODS if m[2]]
ACTIVE_FLOW_METHODS = [m for m in FLOW_METHODS if m[2]]


def pytest_generate_tests(metafunc):
    """Dynamically parametrize test cases from the most recent generated dataset.

    This will provide a single fixture `case_file` that points to one .npz file
    per case directory inside the newest `datasets/dataset_generated_*` folder.
    """

    # If the test function doesn't request `case_file`, nothing to do.
    if "case_file" not in metafunc.fixturenames:
        return
    
    datasets_root = Path("datasets")
    generated = sorted(datasets_root.glob("dataset_generated_*"))
    if not generated:
        # No generated datasets at all -> skip the whole module
        pytest.skip("No generated datasets found under datasets/")

    latest_dataset = generated[-1]
    test_cases = latest_dataset.glob("case_*")
    case_indices = sorted(
        int(re.match(r"case_(\d+)$", p.name).group(1))
        for p in test_cases
        if re.match(r"case_(\d+)$", p.name)
    )
    case_data = {}

    for case_idx in case_indices:
        if case_idx not in SELECTED_TEST_CASES:
            continue

        case_dir = latest_dataset / f"case_{case_idx}"
        case_file_params = []
        case_file_ids = []

        if case_dir.exists():
            npz_files = sorted(case_dir.rglob("*.npz"))
            if npz_files:
                case_file_params = [str(p) for p in npz_files]
                case_file_ids = [p.name for p in npz_files]
        
        case_data[case_idx] = (case_file_params, case_file_ids)

    metafunc.parametrize("case_id,case_file", 
        [(case_idx, cf) for case_idx, (case_file_params, _) in case_data.items() for cf in case_file_params],
        ids=[f"case_{case_idx}-{cf}" for case_idx, (_, case_file_ids) in case_data.items() for cf in case_file_ids]
    )


@pytest.fixture(params=SELECTED_DESCENT_METHODS)
def descent_method(request):
    """Fixture for descent method."""
    return request.param


@pytest.fixture(params=SELECTED_MAX_ITERS)
def max_iters(request):
    """Fixture for maximum iterations."""
    return request.param


@pytest.fixture(params=SELECTED_GRAD_TOLERANCES)
def grad_tolerance(request):
    """Fixture for gradient tolerance."""
    return request.param


@pytest.fixture(params=SELECTED_WIND_METHODS)
def wind_method(request):
    """Fixture for winding method."""
    return request.param


@pytest.fixture(params=SELECTED_SHOULD_GUESS_WIND)
def should_guess_wind(request):    
    """Fixture for whether to guess winding number."""
    return request.param


@pytest.fixture(params=SELECTED_FOURIER_OVERSAMPLES)
def fourier_oversample(request):
    """Fixture for Fourier oversampling factor."""
    return request.param


@pytest.fixture(params=SELECTED_CONVERGENCE_TOLERANCES)
def convergence_tolerance(request):
    """Fixture for convergence tolerance."""
    return request.param


@pytest.fixture(params=SELECTED_MAX_RANDOM_RESTARTS)
def max_random_restarts(request):
    """Fixture for maximum random restarts."""
    return request.param


@pytest.fixture
def case_data(case_file):
    """Fixture to load dataset from case_file."""
    return np.load(case_file)

    
@pytest.fixture
def near_field(case_data):
    """Fixture to extract near_field from case_data."""
    return jnp.asarray(case_data["near_field"], dtype=jnp.complex128)


@pytest.fixture
def far_field(near_field):
    """Fixture to compute far_field from near_field."""
        # build far-field intensity (oversample by specified factor)
    y = jnp.abs(jnp.fft.fft2(near_field, s=jnp.array(jnp.shape(near_field)), norm="ortho"))**2
    return y / (jnp.shape(y)[0]*jnp.shape(y)[1])


@pytest.fixture
def far_field_oversampled(near_field, fourier_oversample):
    """Fixture to compute far_field from near_field."""
        # build far-field intensity (oversample by specified factor)
    y = jnp.abs(jnp.fft.fft2(near_field, s=jnp.array(jnp.shape(near_field))*fourier_oversample, norm="ortho"))**2
    return y / (jnp.shape(y)[0]*jnp.shape(y)[1])


@pytest.fixture
def support_mask(near_field):
    # Use a full mask.
    # TODO: Maybe calculate support from image?
    mask = jnp.ones_like(near_field)
    return mask


@pytest.fixture
def winding_guess(case_data, should_guess_wind):
    """Fixture to extract winding guess from case_data."""
    bright_spot = case_data["spot_center"]
    winding_guess = (int(bright_spot[0]), int(bright_spot[1])) if not should_guess_wind else None
    return winding_guess


@pytest.fixture(params=ACTIVE_INIT_METHODS, ids=[m[0] for m in ACTIVE_INIT_METHODS])
def initial_guess(request, far_field, max_random_restarts):
    """
    This fixture returns the actual x0 vector.
    """
    key = jax.random.PRNGKey(0)

    method_name, method_func, _ = request.param
    
    # Configure retry count based on the method type
    max_attempts = max_random_restarts if method_name == "random" else 1

    def guess_generator():
        current_key = key
        subkey = key
        for _ in range(max_attempts):
            if method_name == "random":
                # For random restarts, we need a fresh key for every iteration
                current_key, subkey = jax.random.split(current_key)

            # TODO: Pass initialization-specific parameters?
            yield method_func(
                near_field_shape=far_field.shape,
                measured_intensity=jnp.abs(far_field),
                random_key=subkey,
            )

    # Return the iterator object so the test can loop over it
    return guess_generator()


@pytest.fixture(params=ACTIVE_FLOW_METHODS, ids=[m[0] for m in ACTIVE_FLOW_METHODS])
def flow_method(request):
    method_name, method_func, _ = request.param
    return method_func


@pytest.fixture
def execution_timer():
    """
    A context manager fixture to measure execution time.
    Usage:
        with execution_timer as t:
            # do work
        print(t.duration)
    """
    class Timer:
        def __enter__(self):
            self.start = time.time()
            self.duration = 0.0
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.duration = time.time() - self.start
            
    return Timer()


@pytest.fixture
def evaluate_convergence(case_id, case_file, convergence_tolerance):
    """
    Returns a FUNCTION that performs error calculation, reporting, 
    and assertion. This cleans up the test body significantly.
    """
    def _evaluate(prediction, ground_truth, timer_obj, metadata=None, output=True):
        # 1. Normalization (Phase Retrieval ambiguity handling)
        # Note: In real phase retrieval, you might need to align the global phase 
        # (e.g., x_out * jnp.exp(-1j * angle)) before comparing. 
        # Here we just normalize norms for simplicity.
        x_out = prediction / jnp.linalg.norm(prediction)
        x_gt = ground_truth / jnp.linalg.norm(ground_truth)

        # 2. Error Calculation (Relative Error)
        err = jnp.linalg.norm(x_out - x_gt) / jnp.linalg.norm(x_gt)
        
        # 3. Reporting
        # We construct a log string from the metadata dict provided
        meta_str = " ".join([f"{k}={v}" for k, v in (metadata or {}).items()])
        duration = timer_obj.duration if timer_obj else 0.0
        
        if output:
            print(f"\n[Case ID={case_id} File={case_file}] {meta_str} | Error={err:.4e} | Time={duration:.4f}s]", flush=True)
        
            # Convergence Check (Assertion)
            # We explicitly cast to float to avoid JAX array boolean ambiguity in some contexts
            assert float(err) < convergence_tolerance, f"Convergence failed. Error {err:.4e} > Tol {convergence_tolerance}"

        return err, (float(err) < convergence_tolerance)

    return _evaluate


def test_phastphase_retrieve(
    case_id: int,
    case_file: str,
    near_field: jnp.ndarray,
    far_field_oversampled: jnp.ndarray,
    support_mask: jnp.ndarray,
    winding_guess: tuple,
    wind_method: int,
    descent_method: int, 
    max_iters: int, 
    grad_tolerance: float, 
    execution_timer,
    evaluate_convergence
):
    """Helper: run retrieve() on one saved near-field object and perform checks.

    Returns the (err, duration) tuple for reporting if needed.
    """
    with execution_timer as t:
        try:
            x_out, val = retrieve(
                far_field_oversampled,
                support_mask,
                max_iters=max_iters,
                descent_method=descent_method,
                grad_tolerance=grad_tolerance,
                wind_method=wind_method,
                winding_guess=winding_guess,
            )
        except Exception as exc:  # pragma: no cover - surface errors as test failures
            pytest.fail(f"retrieve raised an exception: {exc}")

    # This handles normalization, error calc, printing, and asserting.
    evaluate_convergence(
        prediction=x_out,
        ground_truth=near_field,
        timer_obj=t,
        metadata={
            "method": "phastphase",
            "winding_guess": winding_guess,
            "descent_method": descent_method,
            "wind_method": wind_method,
            "max_iters": max_iters,
            "grad_tolerance": grad_tolerance,
        }
    )


def test_gradient_flow_retrieve(
    request,
    case_id: int,
    case_file: str,
    near_field: jnp.ndarray,
    far_field: jnp.ndarray,
    flow_method,
    initial_guess: iter,
    grad_tolerance: float,
    max_iters: int, 
    max_random_restarts: int,
    execution_timer,
    evaluate_convergence
):
    """Helper: run retrieve() on one saved near-field object and perform checks.

    Returns the (err, duration) tuple for reporting if needed.
    """
    init_method_name = request.node.callspec.params["initial_guess"][0]
    flow_method_name = request.node.callspec.params["flow_method"][0]
    
    min_err = float('inf')
    min_prediction = None
    min_t = None

    try:
        for attempt, x0 in enumerate(initial_guess):          
            with execution_timer as t:
                x_out = flow_method(
                    x0,
                    far_field,
                    grad_tolerance=grad_tolerance,
                    iter_limit=max_iters,
                )

            err, success = evaluate_convergence(
                prediction=x_out,
                ground_truth=near_field,
                timer_obj=t,
                output=False,
            )
            if success:
                # Found a solution that converges.
                min_err = err
                min_prediction = x_out
                min_t = t
                break
            else:
                # Capture error and continue to the next guess from the fixture
                if err < min_err:
                    min_err = err
                    min_prediction = x_out
                    min_t = t

    except Exception as exc:  # pragma: no cover - surface errors as test failures
        pytest.fail(f"retrieve raised an exception: {exc}")

    # If we exhaust the generator without returning, the test has failed.
    evaluate_convergence(
        prediction=min_prediction,
        ground_truth=near_field,
        timer_obj=min_t,
        metadata={
            "flow_method": flow_method_name,
            "init_method": init_method_name,
            "attempt": f"{attempt + 1}",
            "max_iters": max_iters,
            "max_random_restarts": max_random_restarts,
            "grad_tolerance": grad_tolerance,
        },
        output=True,
    )
