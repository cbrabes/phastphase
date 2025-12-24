import os
# Limit GPU memory allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # Use 75% of GPU memory

import sys
import itertools
from filelock import FileLock
import random
import pytest
import numpy as np
from pathlib import Path
import re
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from phastphase.retrieval_jax.alternative_methods._gradient_flows import *
from phastphase.retrieval_jax.alternative_methods._initializations import *
from phastphase.retrieval_jax._alternating_projections import *

from tests.test_utils import *

#### CONFIGURE TEST PARAMETERS ####
SELECTED_TEST_CASES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SELECTED_DESCENT_METHODS = [0]
SELECTED_WIND_METHODS = [0]
SELECTED_MAX_ITERS = [1000]
SELECTED_GRAD_TOLERANCES = [1e-8]
SELECTED_SHOULD_GUESS_WIND = [False]
SELECTED_FOURIER_OVERSAMPLES = [2]
SELECTED_CONVERGENCE_TOLERANCES = [5e-4]
SELECTED_MAX_RANDOM_RESTARTS = [1000]
SELECTED_BETAS = [0.36, 0.4, 0.5, 1.17, 1.5]
PERTURBED_INITIAL_GUESS_NOISE_LEVELS = [0.1]
RANDOM_WINDOW_INITIAL_GUESS_WINDOW_SIZES = [100]


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


# (method_name, method_function, is_active?)
PROJECTION_METHODS = [
    ("HIO", HIO, True),
    ("ER", ER, True),
    ("RAAR", RAAR, True),
]


# Filter the list to only include active methods
ACTIVE_INIT_METHODS = [m for m in INITIAL_GUESS_METHODS if m[2]]
ACTIVE_FLOW_METHODS = [m for m in FLOW_METHODS if m[2]]
ACTIVE_PROJECTION_METHODS = [m for m in PROJECTION_METHODS if m[2]]
            

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
    return SELECTED_BETAS


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
    seed = time.time()
    key = jax.random.PRNGKey(seed)

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
    return [
        jnp.pad(
            guess, 
            ((0, guess.shape[0] * (fourier_oversample - 1)), (0, guess.shape[1] * (fourier_oversample - 1))), 
            mode='constant', 
            constant_values=0
        ) for guess in initial_guess
    ]


@pytest.fixture(scope="function")
def perfect_initial_guess(near_field):
    """Fixture that returns the perfect initial guess (the ground truth)."""
    return [near_field]


@pytest.fixture(scope="function")
def padded_perfect_initial_guess(near_field_padded):
    """Fixture that returns the perfect initial guess (the ground truth)."""
    return [near_field_padded]


@pytest.fixture(scope="function", params=PERTURBED_INITIAL_GUESS_NOISE_LEVELS)
def perturbed_initial_guess(request, near_field):
    """Fixture that returns a perturbed version of the ground truth as initial guess."""
    noise_level = request.param
    seed = int(time.time())
    key = jax.random.PRNGKey(seed)
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)

    noise = noise_level * (jax.random.normal(subkey1, near_field.shape) + 1j * jax.random.normal(subkey2, near_field.shape))
    perturbed_guess = near_field + noise
    return [perturbed_guess]


@pytest.fixture(scope="function")
def padded_perturbed_initial_guess(perturbed_initial_guess, fourier_oversample):
    """Fixture that returns a perturbed version of the ground truth as initial guess."""
    return [
        jnp.pad(
                guess, 
                ((0, guess.shape[0] * (fourier_oversample - 1)), (0, guess.shape[1] * (fourier_oversample - 1))), 
            mode='constant', 
            constant_values=0
        ) for guess in perturbed_initial_guess
    ]


@pytest.fixture(scope="function", params=RANDOM_WINDOW_INITIAL_GUESS_WINDOW_SIZES)
def random_window_initial_guess(request, near_field):
    """Fixture that returns a random window initial guess."""
    window_size = request.param
    x_start = random.randint(0, int(near_field.shape[0] / 2))
    y_start = random.randint(0, int(near_field.shape[1] / 2))

    # 1. Initialize array with a complex dtype
    seed = int(time.time())
    key = jax.random.PRNGKey(seed)

    # 2. Split key and generate complex random values
    key, subkey = jax.random.split(key)
    # jax.random.normal supports complex dtypes directly
    # It generates z = x + iy where x, y ~ N(0, 1/sqrt(2))
    random_complex = jax.random.normal(subkey, (window_size, window_size), dtype=jnp.complex64)

    # Wrap to [-pi, pi]
    imag_part = np.angle(random_complex)

    # Combine into a complex-valued array
    complex_map = np.abs(random_complex) * np.exp(1j * imag_part)
    near_field_mod = near_field.at[x_start: x_start + window_size, y_start: y_start + window_size].set(complex_map)

    return [near_field_mod]


@pytest.fixture(scope="function")
def random_initial_guess(near_field, far_field_oversampled):
    """
    This fixture returns the actual x0 vector.
    """
    seed = int(time.time())
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    guess = random_initialization(
        near_field_shape=near_field.shape,
        measured_intensity=far_field_oversampled,
        random_key=subkey,
    )
    return [guess]


@pytest.fixture(scope="function")
def padded_random_initial_guess(near_field_padded, far_field_oversampled):
    """
    This fixture returns the actual x0 vector.
    """
    seed = int(time.time())
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    guess = random_initialization(
        near_field_shape=near_field_padded.shape,
        measured_intensity=far_field_oversampled,
        random_key=subkey,
    )
    return [guess]


@pytest.fixture(scope="session", params=ACTIVE_FLOW_METHODS, ids=[m[0] for m in ACTIVE_FLOW_METHODS])
def flow_method(request):
    method_name, method_func, _ = request.param
    return method_func


@pytest.fixture(scope="session", params=ACTIVE_PROJECTION_METHODS, ids=[m[0] for m in ACTIVE_PROJECTION_METHODS])
def projection_method(request):
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


@pytest.fixture(scope="session", params=[DEFAULT_DB_PATH])
def db_session(request):
    """
    Creates the database connection once for the entire test session.
    """
    # 1. Define a consistent path for the DB
    # Using a shared folder ensures all workers see the same file
    db_path = request.param
    lock_path = f"{db_path}.lock"

    # 2. Use a lock only for the INITIAL creation
    with FileLock(lock_path):
        # The first worker creates the table; others just verify it exists
        init_db = ResultsDatabase(db_path)
        init_db.close()

    # 3. Now every worker opens its OWN connection to the SAME file
    # Because scope="session", this happens once per worker process
    db = ResultsDatabase(db_path)
    
    yield db
    
    # Clean up when the worker finishes its batch of tests
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

        # FFT is invariant to global phase,
        # so we need to cancel relative angle between input and output before comparison. 
        x_out = align_global_phase(x_true=x_gt, x_rec=x_out)

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
            print(f"\n[Case ID={case_id} File={case_file}] {meta_str} | Error={err:.4e} | Time={duration:.4f}s]", flush=True, file=sys.stderr)
        
            # Convergence Check (Assertion)
            # We explicitly cast to float to avoid JAX array boolean ambiguity in some contexts
            assert float(err) < convergence_tolerance, f"Convergence failed. Error {err:.4e} > Tol {convergence_tolerance}"

        return err, (float(err) < convergence_tolerance)

    return _evaluate
