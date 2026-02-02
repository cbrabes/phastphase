from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial
from jax.typing import ArrayLike

from .._linesearch import exact_linesearch_intensity_loss_flat_real
from .._loss_funcs import view_as_flat_real, weighted_intensity_loss
from ._cg_methods import curved_newton_cg_method
from ._trust_region_ncg import cubic_backtracking_linesearch


class _TruncatedNCGState(NamedTuple):
    x_k: jnp.ndarray
    gradient_norm_k: ArrayLike
    step_number: int


def _truncated_ncg_step(
    state: _TruncatedNCGState,
    gradient_function: Callable,
    linesearch_function: Callable,
    max_cg_iters: int,
) -> _TruncatedNCGState:
    """Perform a single step of the truncated Newton conjugate gradient algorithm."""
    grad, hvp = jax.linearize(gradient_function, state.x_k)
    curved_cg_state = curved_newton_cg_method(
        hessvp=hvp, gradient=grad, max_iters=max_cg_iters
    )
    alpha = linesearch_function(
        x=state.x_k, search_direction=curved_cg_state.search_direction
    )

    new_x = state.x_k + alpha * curved_cg_state.search_direction
    new_grad_norm = jnp.linalg.vector_norm(gradient_function(new_x))

    return _TruncatedNCGState(
        x_k=new_x,
        gradient_norm_k=new_grad_norm,
        step_number=state.step_number + 1,
    )


def _truncated_ncg_cond(
    state: _TruncatedNCGState, grad_tolerance: ArrayLike, iter_limit: int
) -> Array:
    return jnp.logical_and(
        jnp.greater(state.gradient_norm_k, grad_tolerance),
        jnp.less(state.step_number, iter_limit),
    )


def truncated_newton_cg(
    cost_function: Callable,
    initial_x: jnp.ndarray,
    iter_limit: int,
    linesarch_function: Callable = cubic_backtracking_linesearch,
    absolute_grad_tolerance: float = 1e-3,
    relative_grad_tolerance: float = 1e-3,
    max_cg_iters: Optional[int] = None,
) -> _TruncatedNCGState:
    grad_function = jax.grad(cost_function)
    linesearch_func = linesarch_function
    if max_cg_iters is None:
        max_cg_iters = jnp.size(initial_x)

    body_func = Partial(
        _truncated_ncg_step,
        gradient_function=grad_function,
        linesearch_function=linesearch_func,
        max_cg_iters=max_cg_iters,
    )
    norm_target = jnp.fmin(
        absolute_grad_tolerance,
        relative_grad_tolerance * jnp.linalg.vector_norm(grad_function(initial_x)),
    )
    cond_func = Partial(
        _truncated_ncg_cond, grad_tolerance=norm_target, iter_limit=iter_limit
    )
    final_state = jax.lax.while_loop(
        cond_func,
        body_func,
        _TruncatedNCGState(
            x_k=initial_x,
            gradient_norm_k=jnp.linalg.vector_norm(grad_function(initial_x)),
            step_number=0,
        ),
    )
    return final_state


def exact_linesearch_tncg(
    initial_x: jnp.ndarray,
    xshape: tuple,
    y: jnp.ndarray,
    phase_ref_point: tuple,
    iter_limit: int,
    absoltue_grad_tolerance: float = 1e-3,
    relative_grad_tolerance: float = 1e-3,
    max_cg_iters: Optional[int] = None,
) -> _TruncatedNCGState:
    """Perform the exact line search for the truncated Newton conjugate gradient algorithm."""
    weights = 1 / jnp.sqrt(y)
    linesearch_func = Partial(
        exact_linesearch_intensity_loss_flat_real,
        y=y,
        phase_ref_point=phase_ref_point,
        xshape=xshape,
        weighting_vector=weights,
    )
    cost_function = Partial(
        weighted_intensity_loss,
        y=y,
        weights=weights,
        phase_ref_point=phase_ref_point,
        x_shape=xshape,
    )
    initial_x = view_as_flat_real(initial_x)
    return truncated_newton_cg(
        cost_function=cost_function,
        initial_x=initial_x,
        iter_limit=iter_limit,
        linesarch_function=linesearch_func,
        absolute_grad_tolerance=absoltue_grad_tolerance,
        relative_grad_tolerance=relative_grad_tolerance,
        max_cg_iters=max_cg_iters,
    )
