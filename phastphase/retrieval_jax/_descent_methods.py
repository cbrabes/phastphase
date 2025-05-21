from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.typing import ArrayLike

from ._linesearch import exact_linesearch_intensity_loss
from ._loss_funcs import view_as_complex, view_as_flat_real, weighted_intensity_loss


class _LBFGSState(NamedTuple):
    opt_state: optax.OptState
    opt_params: jnp.ndarray
    iteration: int
    grad_norm: float


def _lbfgs_cond_func(
    state: _LBFGSState, abs_tol: float, rel_tol: float, max_iters: int, g0: float
) -> jnp.ndarray:
    """Condition function for the LBFGS loop."""
    return jnp.logical_and(
        jnp.less(state.iteration, max_iters),
        jnp.logical_and(
            jnp.greater(state.grad_norm, abs_tol),
            jnp.greater(state.grad_norm, rel_tol * g0),
        ),
    )


def _lbfgs_step(
    state: _LBFGSState,
    cost_function: Callable,
    value_and_grad: Callable,
    lbfgs_solver: optax.GradientTransformationExtraArgs,
) -> _LBFGSState:
    """Perform a single step of the LBFGS optimization algorithm."""
    value, grad = value_and_grad(state.opt_params, state=state.opt_state)
    updates, new_opt_state = lbfgs_solver.update(
        grad,
        state.opt_state,
        state.opt_params,
        value=value,
        grad=grad,
        value_fn=cost_function,
    )
    new_params = optax.apply_updates(state.opt_params, updates)
    return _LBFGSState(
        opt_state=new_opt_state,
        opt_params=new_params,
        iteration=state.iteration + 1,
        grad_norm=jnp.linalg.norm(grad),
    )


def lbfgs_minimize(
    cost_function: Callable,
    x_0: jnp.ndarray,
    max_itertaions: int,
    rtol: float = 1e-5,
    atol: float = 1e-15,
) -> _LBFGSState:
    """Optimize a function using the LBFGS algorithm."""
    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(cost_function)
    g0 = jnp.linalg.norm(value_and_grad(x_0, state=solver.init(x_0))[1])
    state = _LBFGSState(
        opt_state=solver.init(x_0),
        opt_params=x_0,
        iteration=0,
        grad_norm=g0,
    )
    cond_func = jax.tree_util.Partial(
        _lbfgs_cond_func, abs_tol=atol, rel_tol=rtol, max_iters=max_itertaions, g0=g0
    )
    step_func = jax.tree_util.Partial(
        _lbfgs_step,
        cost_function=cost_function,
        value_and_grad=value_and_grad,
        lbfgs_solver=solver,
    )
    final_state = lax.while_loop(cond_func, step_func, state)
    return final_state


def lbfgs_saddlepoint_escape(
    cost_function: Callable,
    x_0: jnp.ndarray,
    max_itertaions: int,
    rtol: float = 1e-5,
    atol: float = 1e-15,
):
    """Optimize a function using a compound LBFGS/ Newton method."""


class _ExactLSGradState(NamedTuple):
    x_k: jnp.ndarray
    grad_k: jnp.ndarray
    step_number: int
    grad_norm_k: ArrayLike


def _exact_linesearch_step(
    state: _ExactLSGradState,
    grad_func: Callable,
    linesearch_function: Callable,
    complex_shape: jnp.ndarray,
) -> _ExactLSGradState:
    """Perform a single step of the exact line search gradient descent algorithm."""
    x_k = view_as_complex(state.x_k, complex_shape)
    grad_k = view_as_complex(state.grad_k, complex_shape)
    step_number = state.step_number

    # Compute the search direction
    search_direction = -grad_k

    # Perform the exact line search
    alpha = linesearch_function(x_k, search_direction)

    # Update the parameters
    x_new = view_as_flat_real(x_k + alpha * search_direction)

    # Compute the new gradient and gradient norm
    new_grad = grad_func(x_new)
    new_grad_norm = jnp.linalg.vector_norm(new_grad)

    return _ExactLSGradState(
        x_k=x_new,
        grad_k=new_grad,
        step_number=step_number + 1,
        grad_norm_k=new_grad_norm,
    )


def _exact_linesearch_cond_func(
    state: _ExactLSGradState, grad_tolerance: float, max_iters: int
) -> jnp.ndarray:
    """Condition function for the exact line search gradient descent loop."""
    return jnp.logical_and(
        jnp.less(state.step_number, max_iters),
        jnp.greater(state.grad_norm_k, grad_tolerance),
    )


def exact_linesearch_gradient_descent(
    x0: jnp.ndarray,
    y: jnp.ndarray,
    grad_tolerance: ArrayLike,
    max_iters: int,
    phase_ref_point: Tuple[int, int],
):
    xshape = x0.shape
    x0 = view_as_flat_real(x0)
    target_intensities = y
    weighting_vector = 1 / jnp.sqrt(y)
    cost_func = jax.tree_util.Partial(
        weighted_intensity_loss,
        y=y,
        weights=weighting_vector,
        phase_ref_point=phase_ref_point,
        x_shape=xshape,
    )
    grad_func = jax.grad(cost_func)
    init_state = _ExactLSGradState(
        x_k=x0,
        grad_k=grad_func(x0),
        step_number=0,
        grad_norm_k=jnp.linalg.vector_norm(grad_func(x0)),
    )
    cond_func = jax.tree_util.Partial(
        _exact_linesearch_cond_func, grad_tolerance=grad_tolerance, max_iters=max_iters
    )
    linesearch_func = jax.tree_util.Partial(
        exact_linesearch_intensity_loss,
        y=target_intensities,
        phase_ref_point=phase_ref_point,
        weighting_vector=1 / jnp.sqrt(target_intensities),
    )
    body_func = jax.tree_util.Partial(
        _exact_linesearch_step,
        grad_func=grad_func,
        linesearch_function=linesearch_func,
        complex_shape=xshape,
    )
    final_state = lax.while_loop(cond_func, body_func, init_state)
    return (final_state.x_k, final_state.step_number)
