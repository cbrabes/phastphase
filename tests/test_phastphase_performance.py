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


SELECTED_TEST_CASES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


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

    # Set default parameters.
    metafunc.parametrize("descent_method", [0])
    metafunc.parametrize("max_iters", [30])
    metafunc.parametrize("grad_tolerance", [1e-8])
    metafunc.parametrize("wind_method", [0])
    metafunc.parametrize("should_guess_wind", [False])
    metafunc.parametrize("fourier_oversample", [2])
    metafunc.parametrize("convergence_tolerance", [1e-3])


def test_phastphase_retrieve(
    case_id: int,
    case_file: str,
    descent_method: int, 
    max_iters: int, 
    grad_tolerance: float, 
    wind_method: int,
    should_guess_wind: bool,
    fourier_oversample: int,
    convergence_tolerance: float,
):
    """Helper: run retrieve() on one saved near-field object and perform checks.

    Returns the (err, duration) tuple for reporting if needed.
    """
    # load dataset
    data = np.load(case_file)
    near_field = jnp.asarray(data["near_field"], dtype=jnp.complex128)
    bright_spot = data["spot_center"]
    winding_guess = (int(bright_spot[0]), int(bright_spot[1])) if not should_guess_wind else None

    # build far-field intensity (oversample by specified factor)
    y = jnp.abs(jnp.fft.fft2(near_field, s=jnp.array(jnp.shape(near_field))*fourier_oversample, norm="ortho"))**2
    y = y / (jnp.shape(y)[0]*jnp.shape(y)[1])

    # Use a full mask.
    # TODO: Maybe calculate support from image?
    mask = jnp.ones_like(near_field)

    start = time.time()
    try:
        x_out, val = retrieve(
            y,
            mask,
            max_iters=max_iters,
            descent_method=descent_method,
            grad_tolerance=grad_tolerance,
            wind_method=wind_method,
            winding_guess=winding_guess,
        )
    except Exception as exc:  # pragma: no cover - surface errors as test failures
        pytest.fail(f"retrieve raised an exception: {exc}")
    duration = time.time() - start

    x_out = x_out / jnp.linalg.vector_norm(x_out)
    near_field = near_field / jnp.linalg.vector_norm(near_field)

    # compute relative error
    err = jnp.linalg.vector_norm(x_out - near_field) / jnp.linalg.vector_norm(near_field)
    
    # Report metrics (pytest will capture these)
    print(f"case_id={case_id} case_file={case_file} descent={descent_method} err={err:.4f} time={duration:.2f}s")

    # ensure routine made progress / returned a reasonable result
    assert err < convergence_tolerance, f"High relative error: {err}"
