from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from ._cg_methods import (
    ITERATION_LIMIT_REACHED,
    _CappedCGState,
    _dot,
    capped_cg_method,
    lobpcg_min_eigpair,
)


class TrustRegionParameters(NamedTuple):
    initial_trust_radius: float
    maximum_trust_radius: float
    step_acceptance_tolerance: float
    gamma_1: float
    gamma_2: float
    psi: float


class _NCGState(NamedTuple):
    x_k: jnp.ndarray
    gradient_norm_k: ArrayLike
    trust_radius: ArrayLike
    step_number: int
    converged: ArrayLike
    convergence_status: ArrayLike
    negative_curvature_direction: jnp.ndarray
    valid_negative_curvature_direction: ArrayLike


def _curvature_condition_check(
    ncg_state: _NCGState,
    gradient: jnp.ndarray,
    candidate_direction: jnp.ndarray,
    curvature_tolerance: float,
    damped_hessvp: Callable,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Function to run if we suspect convergence, or CG stalled"""
    (eigvec, eigval) = lobpcg_min_eigpair(
        damped_hvp=damped_hessvp, example_x=ncg_state.x_k, iter_limit=100
    )
    is_positive_semidefinite = jnp.greater(eigval, -curvature_tolerance / 2)

    dot_sign = jnp.sign(_dot(gradient, eigvec))
    new_direction = -dot_sign * eigvec * ncg_state.trust_radius

    output_direction = lax.select(
        is_positive_semidefinite,
        candidate_direction,
        new_direction,
    )

    return (is_positive_semidefinite, output_direction)


def _no_curvature_condition_check(
    ncg_state: _NCGState,
    gradient: jnp.ndarray,
    candidate_direction: jnp.ndarray,
    curvature_tolerance: float,
) -> Tuple[ArrayLike, ArrayLike]:
    return (False, candidate_direction)


def _is_potential_saddlepoint(
    ncg_state: _NCGState,
    capped_cg_state: _CappedCGState,
    gradient_tolerance: float,
) -> ArrayLike:
    """
    Check for convergence
    """
    return jnp.logical_or(
        jnp.less(ncg_state.gradient_norm_k, gradient_tolerance),
        jnp.equal(capped_cg_state.convergence_status, ITERATION_LIMIT_REACHED),
    )


def _ncg_step(
    state: _NCGState,
    cost_function: Callable,
    gradient_function: Callable,
    trust_params: TrustRegionParameters,
    gradient_tolerance: float,
    curvature_tolerance: float,
    cg_accuracy: float,
    hessian_norm_bound: float,
) -> _NCGState:
    # Use linearize to cache computations for the hessian
    gradient, hessvp = jax.linearize(gradient_function, state.x_k)

    def damped_hessvp(p: jnp.ndarray) -> jnp.ndarray:
        return hessvp(p) + 2 * curvature_tolerance * p

    # Calculate the possible iter limits
    kappa = (hessian_norm_bound + 2 * curvature_tolerance) / 2

    n = jnp.size(state.x_k)
    alt_limit = jnp.astype(
        jnp.ceil(
            0.5 * jnp.sqrt(kappa) * jnp.log(4 * jnp.pow(kappa, 1.5) / cg_accuracy)
        ),
        jnp.int64,
    )
    iter_limit = jnp.fmin(n, alt_limit)

    # Run the CG method
    cg_state = capped_cg_method(
        hessvp=hessvp,
        gradient=gradient,
        damping_factor=curvature_tolerance,
        trust_radius=state.trust_radius,
        relative_accuracy=cg_accuracy,
        iter_limit=iter_limit,
    )

    condition_check = jax.tree_util.Partial(
        _curvature_condition_check, damped_hessvp=damped_hessvp
    )
    maybe_saddlepoint = _is_potential_saddlepoint(state, cg_state, gradient_tolerance)
    (is_psd, search_direction) = lax.cond(
        maybe_saddlepoint,
        condition_check,
        _no_curvature_condition_check,
        state,
        gradient,
        cg_state.search_direction,
        curvature_tolerance,
    )
    # Take the CG Search Direction

    # Compute the actual reduction
    actual_reduction = cost_function(state.x_k) - cost_function(
        state.x_k + search_direction
    )

    # Compute the reduction predicted by the quadratic model
    predicted_reduction = _dot(gradient, search_direction)
    predicted_reduction += 0.5 * _dot(search_direction, hessvp(search_direction))

    # Calculate the ratio of the actual reduction to the predicted reduction
    reduction_ratio = -actual_reduction / predicted_reduction

    # If the quadratic model predicts the decrease sufficiently accurately, accept the step
    accept_step = jnp.greater(reduction_ratio, trust_params.step_acceptance_tolerance)

    x_new = lax.select(accept_step, state.x_k + search_direction, state.x_k)
    larger_trust_radius = jnp.fmin(
        trust_params.gamma_2 * state.trust_radius,
        trust_params.maximum_trust_radius,
    )

    search_norm = jnp.linalg.vector_norm(search_direction)

    # If the step direciton is large enough, increase the trust radius, capping with the maximum radius
    increase_trust_radius = jnp.greater(
        search_norm, trust_params.psi * state.trust_radius
    )

    # Compute the new trust radius.
    # If we accept the step, then increase the trust radius if the step is large enough, otherwise keep the same trust radius.
    # If we reject the step, then set the trust radius to gamma_1 times the search norm
    new_trust_radius = lax.select(
        accept_step,
        lax.select(
            increase_trust_radius,
            larger_trust_radius,
            state.trust_radius,
        ),
        trust_params.gamma_1 * search_norm,
    )
    converged = jnp.logical_and(
        is_psd, jnp.less_equal(state.gradient_norm_k, gradient_tolerance)
    )

    return _NCGState(
        x_k=x_new,
        gradient_norm_k=jnp.linalg.vector_norm(gradient),
        trust_radius=new_trust_radius,
        step_number=state.step_number + 1,
        converged=converged,
        convergence_status=cg_state.convergence_status,
        negative_curvature_direction=cg_state.search_direction,
        valid_negative_curvature_direction=is_psd,
    )


def trust_region_ncg_method(
    cost_function: Callable,
    initial_x: jnp.ndarray,
    gradient_tolerance: float,
    curvature_tolerance: float,
    cg_accuracy: float,
    hessian_norm_bound: float,
    trust_region_parameters: TrustRegionParameters,
):
    return None


def dummy():
    x = curvature_check
    y = capped_cg_method
    return None
