import os
# Limit GPU memory allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # Use 75% of GPU memory

import random
import time
import datetime
import sqlite3
import json
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
from phastphase.retrieval_jax._alternating_projections import *


SELECTED_TEST_CASES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SELECTED_DESCENT_METHODS = [0]
SELECTED_WIND_METHODS = [0]
SELECTED_MAX_ITERS = [1000]
SELECTED_GRAD_TOLERANCES = [1e-8]
SELECTED_SHOULD_GUESS_WIND = [False]
SELECTED_FOURIER_OVERSAMPLES = [2]
SELECTED_CONVERGENCE_TOLERANCES = [5e-4]
SELECTED_MAX_RANDOM_RESTARTS = [1000]
SELECTED_BETAS = [0.36, 0.4, 0.5, 1.17]

RANDOM_GENERATOR = np.random.default_rng(0)


# (method_name, method_function, is_active?)
INITIAL_GUESS_METHODS = [
    ("random", random_initialization, False),
    ("spectral", spectral_initialization, True),
    ("trunc_spectral", truncated_spectral_initialization, True),
    ("ortho", orthogonality_promoting_initialization, True),
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


class ResultsDatabase:
    def __init__(self, db_path="tests/phase_retrieval_results.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()
    
    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            case_id INTEGER,
            case_file TEXT,
            method_name TEXT,
            init_method_name TEXT,
            num_attempts INTEGER,
            error REAL,
            grad_tolerance REAL,
            max_iters INTEGER,
            fourier_oversample INTEGER,
            convergence_tolerance REAL,
            compute_duration REAL,
            near_field_md5 TEXT,
            metadata JSON
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def save_result(
        self,
        case_id: int, 
        case_file: str, 
        method_name: str, 
        init_method_name: str, 
        num_attempts: int, 
        error: float, 
        compute_duration: float, 
        grad_tolerance: float, 
        convergence_tolerance: float,
        max_iters: int,
        fourier_oversample: int,
        near_field_md5: str,
        metadata: dict
    ):
        query = """
            
        INSERT INTO results (timestamp, case_id, case_file, method_name, init_method_name, num_attempts, error, grad_tolerance, max_iters, fourier_oversample, convergence_tolerance, compute_duration, near_field_md5, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.execute(query, (
            datetime.datetime.now().isoformat(),
            case_id, 
            case_file, 
            method_name, 
            init_method_name, 
            num_attempts, 
            float(error), 
            float(grad_tolerance), 
            max_iters, 
            fourier_oversample, 
            float(convergence_tolerance), 
            float(compute_duration), 
            near_field_md5, 
            json.dumps(metadata)
        ))
        self.conn.commit()

    def close(self):
        self.conn.close()


class CachedGenerator:
    def __init__(self, iterable):
        self.iterable = iter(iterable)
        self.cache = []

    def __iter__(self):
        # First, yield everything we've already computed
        yield from self.cache
        
        # Then, continue computing new values and caching them
        for item in self.iterable:
            self.cache.append(item)
            yield item

            
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
        ids=[f"case_{case_idx}-{cf}" for case_idx, (_, case_file_ids) in case_data.items() for cf in case_file_ids],
        scope="session"
    )


@pytest.fixture(scope="session", params=SELECTED_DESCENT_METHODS)
def descent_method(request):
    """Fixture for descent method."""
    return request.param


@pytest.fixture(scope="session", params=SELECTED_MAX_ITERS)
def max_iters(request):
    """Fixture for maximum iterations."""
    return request.param


@pytest.fixture(scope="session", params=SELECTED_GRAD_TOLERANCES)
def grad_tolerance(request):
    """Fixture for gradient tolerance."""
    return request.param


@pytest.fixture(scope="session", params=SELECTED_WIND_METHODS)
def wind_method(request):
    """Fixture for winding method."""
    return request.param


@pytest.fixture(scope="session", params=SELECTED_SHOULD_GUESS_WIND)
def should_guess_wind(request):    
    """Fixture for whether to guess winding number."""
    return request.param


@pytest.fixture(scope="session", params=SELECTED_FOURIER_OVERSAMPLES)
def fourier_oversample(request):
    """Fixture for Fourier oversampling factor."""
    return request.param


@pytest.fixture(scope="session", params=SELECTED_CONVERGENCE_TOLERANCES)
def convergence_tolerance(request):
    """Fixture for convergence tolerance."""
    return request.param


@pytest.fixture(scope="session", params=SELECTED_MAX_RANDOM_RESTARTS)
def max_random_restarts(request):
    """Fixture for maximum random restarts."""
    return request.param    


@pytest.fixture(scope="session")
def betas():
    return iter(SELECTED_BETAS)


@pytest.fixture(scope="session")
def case_data(case_id, case_file):
    """Fixture to load dataset from case_file."""
    return np.load(case_file)

    
@pytest.fixture(scope="session")
def near_field(case_data):
    """Fixture to extract near_field from case_data."""
    return jnp.asarray(case_data["near_field"], dtype=jnp.complex128)


@pytest.fixture(scope="session")
def near_field_padded(near_field, fourier_oversample):
    """Fixture to extract near_field from case_data."""
    padded_near_field = jnp.pad(
        near_field, 
        ((0, near_field.shape[0] * (fourier_oversample - 1)), (0, near_field.shape[1] * (fourier_oversample - 1))), 
        mode='constant', 
        constant_values=0
    )

    return padded_near_field


@pytest.fixture(scope="session")
def near_field_md5(case_data):
    """Fixture to extract near_field from case_data."""
    return str(case_data["near_field_md5"])


@pytest.fixture(scope="session")
def far_field(near_field):
    """Fixture to compute far_field from near_field."""
        # build far-field intensity (oversample by specified factor)
    y = jnp.abs(jnp.fft.fft2(near_field, s=jnp.array(jnp.shape(near_field)), norm="ortho"))**2
    return y / (jnp.shape(y)[0]*jnp.shape(y)[1])


@pytest.fixture(scope="session")
def far_field_oversampled(near_field, fourier_oversample):
    """Fixture to compute far_field from near_field."""
        # build far-field intensity (oversample by specified factor)
    y = jnp.abs(jnp.fft.fft2(near_field, s=jnp.array(jnp.shape(near_field))*fourier_oversample, norm="ortho"))**2
    return y / (jnp.shape(y)[0]*jnp.shape(y)[1])


@pytest.fixture(scope="session")
def support_mask(near_field):
    # Use a full mask.
    # TODO: Maybe calculate support from image?
    mask = jnp.ones_like(near_field)
    return mask


@pytest.fixture(scope="session")
def padded_support_mask(support_mask, fourier_oversample):
    padded_mask = jnp.pad(
        support_mask, 
        ((0, support_mask.shape[0] * (fourier_oversample - 1)), (0, support_mask.shape[1] * (fourier_oversample - 1))), 
        mode='constant', 
        constant_values=0
    )

    return padded_mask


@pytest.fixture(scope="session")
def winding_guess(case_data, should_guess_wind):
    """Fixture to extract winding guess from case_data."""
    bright_spot = case_data["spot_center"]
    winding_guess = (int(bright_spot[0]), int(bright_spot[1])) if not should_guess_wind else None
    return winding_guess


@pytest.fixture(scope="session", params=ACTIVE_INIT_METHODS, ids=[m[0] for m in ACTIVE_INIT_METHODS])
def initial_guess_data(request, near_field, far_field_oversampled, max_random_restarts):
    """
    This fixture returns the actual x0 vector.
    """
    # FIXME: We will always be generating the same sequence of random numbers!
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
                near_field_shape=near_field.shape,
                measured_intensity=far_field_oversampled,
                random_key=subkey,
            )

    # Return the iterator object so the test can loop over it
    return CachedGenerator(guess_generator())


@pytest.fixture(scope="function")
def initial_guess(initial_guess_data):
    """
    This fixture returns the actual x0 vector.
    """
    return iter(initial_guess_data)


@pytest.fixture(scope="function")
def padded_initial_guess(initial_guess, fourier_oversample):
    def padded_guess_generator():
        for guess in initial_guess:
            yield jnp.pad(
                guess, 
                ((0, guess.shape[0] * (fourier_oversample - 1)), (0, guess.shape[1] * (fourier_oversample - 1))), 
            mode='constant', 
            constant_values=0
        )

    return padded_guess_generator()


@pytest.fixture(scope="session", params=ACTIVE_FLOW_METHODS, ids=[m[0] for m in ACTIVE_FLOW_METHODS])
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


@pytest.fixture(scope="session")
def db_session():
    """
    Creates the database connection once for the entire test session.
    """
    db = ResultsDatabase()
    yield db
    db.close()


@pytest.fixture
def evaluate_convergence(
    case_id, 
    case_file, 
    convergence_tolerance, 
    max_iters, 
    grad_tolerance, 
    fourier_oversample, 
    near_field_md5,
    db_session
):
    """
    Returns a FUNCTION that performs error calculation, reporting, 
    and assertion. This cleans up the test body significantly.
    """
    def _evaluate(
        prediction, 
        ground_truth, 
        timer_obj, 
        method_name, 
        init_method_name="", 
        num_attempts=1,
        metadata=None, 
        output=True,
        save_to_db=True,
    ):
        # Normalization (Phase Retrieval ambiguity handling)
        # Note: In real phase retrieval, you might need to align the global phase 
        # (e.g., x_out * jnp.exp(-1j * angle)) before comparing. 
        # Here we just normalize norms for simplicity.
        x_out = prediction / jnp.linalg.norm(prediction)
        x_gt = ground_truth / jnp.linalg.norm(ground_truth)

        x_out = x_out / jnp.sign(x_out[0,0]) # Fix global phase ambiguity for error calc
        x_gt = x_gt / jnp.sign(x_gt[0,0]) # Fix global phase ambiguity for error calc

        # Error Calculation (Relative Error)
        err = jnp.linalg.norm(x_out - x_gt) / jnp.linalg.norm(x_gt)
        
        # Reporting
        # We construct a log string from the metadata dict provided
        meta_str = " ".join([f"{k}={v}" for k, v in (metadata or {}).items()])
        duration = timer_obj.duration if timer_obj else 0.0
        
        # Save to Database
        if save_to_db:
            db_session.save_result(
                case_id=case_id,
                case_file=case_file,
                method_name=method_name,
                init_method_name=init_method_name,
                num_attempts=num_attempts,
                error=float(err),
                compute_duration=float(duration),
                grad_tolerance=float(grad_tolerance),
                convergence_tolerance=float(convergence_tolerance),
                max_iters=int(max_iters),
                fourier_oversample=int(fourier_oversample),
                near_field_md5=near_field_md5,
                metadata=metadata or {}
            )

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
            return

    # This handles normalization, error calc, printing, and asserting.
    evaluate_convergence(
        prediction=x_out,
        ground_truth=near_field,
        timer_obj=t,
        method_name="phastphase",
        metadata={
            "winding_guess": winding_guess,
            "descent_method": descent_method,
            "wind_method": wind_method,
        }
    )


def test_gradient_flow_retrieve(
    request,
    case_id: int,
    case_file: str,
    near_field: jnp.ndarray,
    far_field_oversampled: jnp.ndarray,
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
                    far_field_oversampled,
                    grad_tolerance=grad_tolerance,
                    iter_limit=max_iters,
                )

            err, success = evaluate_convergence(
                prediction=x_out,
                ground_truth=near_field,
                timer_obj=t,
                method_name=flow_method_name,
                init_method_name=init_method_name,
                num_attempts=attempt + 1,
                metadata={
                    "max_random_restarts": max_random_restarts,
                },
                output=False,
                save_to_db=False,
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
        return

    # If we exhaust the generator without returning, the test has failed.
    evaluate_convergence(
        prediction=min_prediction,
        ground_truth=near_field,
        timer_obj=min_t,
        method_name=flow_method_name,
        init_method_name=init_method_name,
        num_attempts=attempt + 1,
        metadata={
            "max_random_restarts": max_random_restarts,
        },
        output=True,
        save_to_db=True,
    )


def test_gradient_flow_sanity(
    request,
    case_id: int,
    case_file: str,
    near_field: jnp.ndarray,
    far_field_oversampled: jnp.ndarray,
    flow_method,
    # initial_guess: iter,
    grad_tolerance: float,
    max_iters: int, 
    max_random_restarts: int,
    execution_timer,
    evaluate_convergence
):
    """Helper: run retrieve() on one saved near-field object and perform checks.

    Returns the (err, duration) tuple for reporting if needed.
    """
    init_method_name = "sanity" #request.node.callspec.params["initial_guess"][0]
    flow_method_name = request.node.callspec.params["flow_method"][0]
    
    min_err = float('inf')
    min_prediction = None
    min_t = None

    x_start = random.randint(0, int(near_field.shape[0] / 2))
    y_start = random.randint(0, int(near_field.shape[1] / 2))
    k = 300
    # 1. Initialize array with a complex dtype
    key = jax.random.PRNGKey(0)

    # 2. Split key and generate complex random values
    key, subkey = jax.random.split(key)
    # jax.random.normal supports complex dtypes directly
    # It generates z = x + iy where x, y ~ N(0, 1/sqrt(2))
    random_complex = jax.random.normal(subkey, (k, k), dtype=jnp.complex64)

    # Wrap to [-pi, pi]
    imag_part = np.angle(random_complex)

    # Combine into a complex-valued array
    complex_map = np.abs(random_complex) * np.exp(1j * imag_part)
    
    near_field_mod = near_field.at[x_start: x_start + k, y_start: y_start + k].set(complex_map)
    initial_guess = [near_field_mod]  # Sanity check with perfect data
    # far_field_oversampled = near_field  # Sanity check with perfect data

    try:
        for attempt, x0 in enumerate(initial_guess):          
            with execution_timer as t:
                x_out = flow_method(
                    x0,
                    far_field_oversampled,
                    grad_tolerance=grad_tolerance,
                    iter_limit=max_iters,
                )

            err, success = evaluate_convergence(
                prediction=x_out,
                ground_truth=near_field,
                timer_obj=t,
                method_name=flow_method_name,
                init_method_name=init_method_name,
                num_attempts=attempt + 1,
                metadata={
                    "max_random_restarts": max_random_restarts,
                },
                output=False,
                save_to_db=False,
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
        return

    # If we exhaust the generator without returning, the test has failed.
    evaluate_convergence(
        prediction=min_prediction,
        ground_truth=near_field,
        timer_obj=min_t,
        method_name=flow_method_name,
        init_method_name=init_method_name,
        num_attempts=attempt + 1,
        metadata={
            "max_random_restarts": max_random_restarts,
        },
        output=True,
        save_to_db=False,
    )


def test_HIO_retrieve(
    request,
    case_id: int,
    case_file: str,
    near_field: jnp.ndarray,
    near_field_padded: jnp.ndarray,
    far_field_oversampled: jnp.ndarray,
    padded_support_mask: jnp.ndarray,
    padded_initial_guess: iter,
    betas: iter,
    grad_tolerance: float,
    max_iters: int, 
    max_random_restarts: int,
    execution_timer,
    evaluate_convergence
):
    """Helper: run retrieve() on one saved near-field object and perform checks.

    Returns the (err, duration) tuple for reporting if needed.
    """
    init_method_name = request.node.callspec.params["initial_guess_data"][0]
    
    min_err = float('inf')
    min_prediction = None
    min_t = None

    try:
        for attempt, (x0, beta) in enumerate(zip(padded_initial_guess, betas)):          
            with execution_timer as t:
                # import pdb; pdb.set_trace()
                # x0 = near_field_padded
                x_out, num_iterations, residual = HIO(
                    x0=x0,
                    mask=padded_support_mask,
                    y=far_field_oversampled,
                    beta=beta,
                    tolerance=grad_tolerance,
                    max_iters=max_iters,
                )
                import pdb; pdb.set_trace()
                x_out_sliced = x_out[:near_field.shape[0], :near_field.shape[1]]

            err, success = evaluate_convergence(
                prediction=x_out_sliced,
                ground_truth=near_field,
                timer_obj=t,
                method_name="HIO",
                init_method_name=init_method_name,
                num_attempts=attempt + 1,
                metadata={
                    "max_random_restarts": max_random_restarts,
                },
                output=False,
                save_to_db=False,
            )
            if success:
                # Found a solution that converges.
                min_err = err
                min_prediction = x_out_sliced
                min_t = t
                break
            else:
                # Capture error and continue to the next guess from the fixture
                if err < min_err:
                    min_err = err
                    min_prediction = x_out_sliced
                    min_t = t

    except Exception as exc:  # pragma: no cover - surface errors as test failures
        pytest.fail(f"retrieve raised an exception: {exc}")

    # If we exhaust the generator without returning, the test has failed.
    evaluate_convergence(
        prediction=min_prediction,
        ground_truth=near_field,
        timer_obj=min_t,
        method_name="HIO",
        init_method_name=init_method_name,
        num_attempts=attempt + 1,
        metadata={
            "max_random_restarts": max_random_restarts,
        },
        output=True,
        save_to_db=False,
    )
