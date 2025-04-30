from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from ._cg_methods import (
    INTERIOR_SOLUTION,
    ITERATION_LIMIT_REACHED,
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


class _TrustNCGState(NamedTuple):
    x_k: jnp.ndarray
    gradient_norm_k: ArrayLike
    trust_radius: ArrayLike
    step_number: int
    likely_psd: bool


def _curvature_condition_check(
    ncg_state: _TrustNCGState,
    gradient: jnp.ndarray,
    candidate_direction: jnp.ndarray,
    curvature_tolerance: float,
    hessvp: Callable,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Function to run if we suspect convergence, or CG stalled"""
    (eigvec, eigval) = lobpcg_min_eigpair(
        hvp=hessvp, example_x=ncg_state.x_k, iter_limit=100
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
    ncg_state: _TrustNCGState,
    gradient: jnp.ndarray,
    candidate_direction: jnp.ndarray,
    curvature_tolerance: float,
) -> Tuple[ArrayLike, ArrayLike]:
    return (False, candidate_direction)


def cubic_backtracking_linesearch(
    fun: Callable,
    x0: jnp.ndarray,
    direction: jnp.ndarray,
    acceptance_tolerance: ArrayLike = 0.25,
    line_search_decrease_factor: float = 0.5,
):
    """
    Cubic backtracking linesearch for trust region methods.
    Finds alpha such that f(x0 + alpha * direction) <= f(x0) - tolerance * alpha^3 * ||direction||^3
    Returns x0 + alpha * direction
    """
    f0 = fun(x0)

    def _linesearch_cond_func(alpha: ArrayLike) -> ArrayLike:
        return jnp.greater(
            fun(x0 + alpha * direction),
            f0
            - acceptance_tolerance
            * alpha**3
            * jnp.linalg.vector_norm(direction, ord=2) ** 3,
        )

    def _linesearch_body_func(alpha: float) -> float:
        return alpha * line_search_decrease_factor

    alpha = lax.while_loop(_linesearch_cond_func, _linesearch_body_func, 1.0)
    return x0 + alpha * direction


def saddle_check_and_escape(
    ncg_state: _TrustNCGState,
    cost_func: Callable,
    gradient: jnp.ndarray,
    candidate_point: jnp.ndarray,
    curvature_tolerance: float,
    hessvp: Callable,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Function to run if we suspect convergence, or CG stalled"""
    (eigvec, eigval) = lobpcg_min_eigpair(
        hvp=hessvp, example_x=ncg_state.x_k, iter_limit=100
    )
    is_positive_semidefinite = jnp.greater(eigval, -curvature_tolerance / 2)

    dot_sign = jnp.sign(_dot(gradient, eigvec))
    search_direction = -dot_sign * eigvec * jnp.abs(eigval)

    new_point = cubic_backtracking_linesearch(
        cost_func, ncg_state.x_k, search_direction
    )

    output_point = lax.select(
        is_positive_semidefinite,
        candidate_point,
        new_point,
    )

    return (is_positive_semidefinite, output_point)


def no_saddle_check_and_escape(
    ncg_state: _TrustNCGState,
    gradient: jnp.ndarray,
    candidate_point: jnp.ndarray,
    curvature_tolerance: float,
) -> Tuple[ArrayLike, ArrayLike]:
    return (False, candidate_point)


"""
Logic for newton conjugate gradient trust region.

We use the unregularized version, as regularization severely hampers performance in practice.
:

- Run CG method  and get output CG State
if gradient is small and (cg stalls or interior solution found):
    we attempt to find a negative curvature direction using LOBPCG.
    If we find it, we use a backtracking line search to step to the next point.

else:
    use the CG search direction as the search direction.
    
If not converged:
    run standard trust region step logic with search direction. 






"""


def _trust_ncg_step(
    state: _TrustNCGState,
    cost_function: Callable,
    gradient_function: Callable,
    trust_params: TrustRegionParameters,
    gradient_tolerance: float,
    curvature_tolerance: float,
    cg_accuracy: float,
    hessian_norm_bound: float,
) -> _TrustNCGState:
    # Use linearize to cache computations for the hessian
    gradient, hessvp = jax.linearize(gradient_function, state.x_k)

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
        trust_radius=state.trust_radius,
        relative_accuracy=cg_accuracy,
        iter_limit=iter_limit,
    )

    search_direction = cg_state.search_direction
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

    maybe_saddle_point = jnp.logical_and(
        jnp.less(state.gradient_norm_k, gradient_tolerance),
        jnp.logical_or(
            jnp.equal(cg_state.convergence_status, ITERATION_LIMIT_REACHED),
            jnp.equal(cg_state.convergence_status, INTERIOR_SOLUTION),
        ),
    )
    saddle_check = jax.tree_util.Partial(
        saddle_check_and_escape, hessvp=hessvp, cost_function=cost_function
    )
    (likely_psd, x_new) = lax.cond(
        maybe_saddle_point,
        saddle_check,
        no_saddle_check_and_escape,
        state,
        gradient,
        x_new,
        curvature_tolerance,
    )

    return _TrustNCGState(
        x_k=x_new,
        gradient_norm_k=jnp.linalg.vector_norm(gradient),
        trust_radius=new_trust_radius,
        step_number=state.step_number + 1,
        likely_psd=likely_psd,
    )


def trust_region_ncg_method(
    cost_function: Callable,
    initial_x: jnp.ndarray,
    gradient_tolerance: float,
    curvature_tolerance: float,
    cg_accuracy: float,
    hessian_norm_bound: float,
    trust_region_parameters: TrustRegionParameters,
    outer_iter_limit: int = 1000,
):
    _ncg_body_func = jax.tree_util.Partial(
        _trust_ncg_step,
        cost_function=cost_function,
        gradient_function=jax.grad(cost_function),
        trust_params=trust_region_parameters,
        gradient_tolerance=gradient_tolerance,
        curvature_tolerance=curvature_tolerance,
        cg_accuracy=cg_accuracy,
        hessian_norm_bound=hessian_norm_bound,
    )

    def _ncg_cond_func(state: _TrustNCGState) -> ArrayLike:
        local_minimum = jnp.logical_and(
            jnp.less(state.gradient_norm_k, gradient_tolerance), state.likely_psd
        )
        return jnp.logical_and(
            jnp.logical_not(local_minimum),
            jnp.less(state.step_number, outer_iter_limit),
        )

    init_state = _TrustNCGState(
        x_k=initial_x,
        gradient_norm_k=jnp.inf,
        trust_radius=trust_region_parameters.initial_trust_radius,
        step_number=0,
        likely_psd=False,
    )

    final_state = lax.while_loop(_ncg_cond_func, _ncg_body_func, init_state)
    return final_state
