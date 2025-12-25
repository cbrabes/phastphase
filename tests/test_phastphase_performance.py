import itertools
import random
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from phastphase.retrieval_jax import retrieve
from phastphase.retrieval_jax._alternating_projections import *

from tests.test_utils import *


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
        },
        output=True,
        save_to_db=True,
        ensure_success=True
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
    init_method_name = request.node.callspec.params["initial_guess_data"][0]
    flow_method_name = request.node.callspec.params["flow_method"][0]
    
    min_err = float('inf')
    min_prediction = None
    min_t = None
    attempt = 0

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
                ensure_success=False
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
        ensure_success=True
    )


def test_gradient_flow_sanity(
    request,
    case_id: int,
    case_file: str,
    near_field: jnp.ndarray,
    far_field_oversampled: jnp.ndarray,
    perturbed_initial_guess: iter,
    flow_method,
    grad_tolerance: float,
    max_iters: int, 
    max_random_restarts: int,
    execution_timer,
    evaluate_convergence
):
    """Helper: run retrieve() on one saved near-field object and perform checks.

    Returns the (err, duration) tuple for reporting if needed.
    """
    init_method_name = "sanity"
    flow_method_name = request.node.callspec.params["flow_method"][0]
    
    min_err = float('inf')
    min_prediction = None
    min_t = None
    attempt = 0

    try:
        for attempt, x0 in enumerate(perturbed_initial_guess):          
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
                ensure_success=False
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
        ensure_success=True
    )


def test_alternating_projection_retrieve(
    request,
    case_id: int,
    case_file: str,
    near_field: jnp.ndarray,
    far_field_oversampled: jnp.ndarray,
    padded_support_mask: jnp.ndarray,
    padded_random_initial_guess: list,
    # padded_perturbed_initial_guess: list,
    projection_method,
    betas: list,
    grad_tolerance: float,
    max_iters: int, 
    execution_timer,
    evaluate_convergence
):
    """Helper: run retrieve() on one saved near-field object and perform checks.

    Returns the (err, duration) tuple for reporting if needed.
    """    
    projection_method_name = request.node.callspec.params["projection_method"][0]
    
    min_err = float('inf')
    min_prediction = None
    min_t = None
    min_beta = None
    min_residual = None
    min_num_iterations = None

    try:
        for attempt, (x0, beta) in enumerate(itertools.product(padded_random_initial_guess, betas)):  
        # for attempt, (x0, beta) in enumerate(itertools.product(padded_perturbed_initial_guess, betas)):          
            with execution_timer as t:
                x_out, num_iterations, residual = projection_method(
                    x0=x0,
                    mask=padded_support_mask,
                    y=far_field_oversampled,
                    beta=beta,
                    tolerance=grad_tolerance,
                    max_iters=max_iters,
                )
                x_out_sliced = x_out.at[:near_field.shape[0], :near_field.shape[1]].get()

            err, success = evaluate_convergence(
                prediction=x_out_sliced,
                ground_truth=near_field,
                timer_obj=t,
                method_name=projection_method_name,
                init_method_name="random",
                num_attempts=attempt + 1,
                metadata={
                    "num_iterations": int(num_iterations),
                    "residual": float(residual),
                    "beta": beta,
                },
                output=True,
                save_to_db=True,
                ensure_success=False
            )
            if success:
                # Found a solution that converges.
                min_err = err
                min_prediction = x_out_sliced
                min_t = t
                min_beta = beta
                min_residual = residual
                min_num_iterations = num_iterations
                break
            else:
                # Capture error and continue to the next guess from the fixture
                if err < min_err:
                    min_err = err
                    min_prediction = x_out_sliced
                    min_t = t
                    min_beta = beta
                    min_residual = residual
                    min_num_iterations = num_iterations

    except Exception as exc:  # pragma: no cover - surface errors as test failures
        pytest.fail(f"retrieve raised an exception: {exc}")

    # If we exhaust the generator without returning, the test has failed.
    evaluate_convergence(
        prediction=min_prediction,
        ground_truth=near_field,
        timer_obj=min_t,
        method_name=projection_method_name,
        init_method_name="random",
        num_attempts=attempt + 1,
        metadata={
            "num_iterations": int(min_num_iterations),
            "residual": float(min_residual),
            "beta": min_beta,
        },
        output=True,
        save_to_db=False,
        ensure_success=True
    )
