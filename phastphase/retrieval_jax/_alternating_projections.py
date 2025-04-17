from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ._loss_funcs import normalized_intensity_loss


class _HIOState(NamedTuple):
    x: jnp.ndarray
    iteration: int
    residual: jnp.ndarray


def _HIO_projection_step(
    state: _HIOState, mask: jnp.ndarray, y: jnp.ndarray, beta: float
) -> _HIOState:
    """Perform the HIO projection step.

    Args:
        x (jax.typing.ArrayLike): The input array to project.
        beta (float): The beta parameter for the HIO projection.

    Returns:
        jax.typing.ArrayLike: The projected array.
    """
    # Apply the HIO projection step
    y_c = jnp.fft.fft2(state.x)
    y_c = jnp.sign(y_c) * jnp.sqrt(y)
    x_new = jnp.fft.ifft2(y_c)
    x_new = mask * (x_new) + (1 - mask) * (state.x - beta * x_new)
    residual = normalized_intensity_loss(mask * x_new, y)
    return _HIOState(x=x_new, iteration=state.iteration + 1, residual=residual)


def _HIO_cond_func(state: _HIOState, tolerance: float, max_iters: int) -> jnp.ndarray:
    """Condition function for the HIO loop.

    Args:
        state (_HIOState): The current state of the HIO algorithm.
        tolerance (float): The tolerance for convergence.
        max_iters (int): The maximum number of iterations.

    Returns:
        bool: True if the loop should continue, False otherwise.
    """
    return jnp.logical_and(
        jnp.less(state.iteration, max_iters), jnp.greater(state.residual, tolerance)
    )


def HIO(
    x0: jnp.ndarray,
    mask: jnp.ndarray,
    y: jnp.ndarray,
    tolerance: float = 1e-3,
    max_iters: int = 100,
    beta: float = 0.9,
) -> _HIOState:
    """Perform the Hybrid Input-Output (HIO) algorithm."""
    HIO_step_func = jax.tree_util.Partial(
        _HIO_projection_step, mask=mask, y=y, beta=beta
    )
    HIO_cond_func = jax.tree_util.Partial(
        _HIO_cond_func, tolerance=tolerance, max_iters=max_iters
    )
    initial_state = _HIOState(
        x=x0,
        iteration=0,
        residual=jnp.linalg.norm(jnp.square(jnp.abs(jnp.fft.fft2(mask * x0))) - y),
    )
    final_state = lax.while_loop(HIO_cond_func, HIO_step_func, initial_state)

    return final_state


def _damped_ER_projection_step(
    state: _HIOState,
    mask: jnp.ndarray,
    y: jnp.ndarray,
    near_field_damping: float,
    far_field_damping: float,
) -> _HIOState:
    """Perform the Damped Error Reduction (ER) projection step.

    Args:
        state (_DampedERState): The current state of the ER algorithm.
        mask (jax.typing.ArrayLike): The mask to apply.
        y (jax.typing.ArrayLike): The target array.
        near_field_damping (float): The damping factor for the near field.
        far_field_damping (float): The damping factor for the far field.

    Returns:
        _DampedERState: The updated state after applying the ER projection step.
    """
    # Apply the ER projection step
    y_c = jnp.fft.fft2(state.x)
    y_c = far_field_damping * y_c + (1 - far_field_damping) * jnp.sqrt(y) * jnp.sign(
        y_c
    )  # enforce magnitude
    x_new = jnp.fft.ifft2(y_c)

    # Apply damping factors
    x_new = mask * x_new + (1 - mask) * (near_field_damping * x_new)

    residual = normalized_intensity_loss(mask * x_new, y)

    return _HIOState(x=x_new, iteration=state.iteration + 1, residual=residual)


def damped_ER(
    x0: jnp.ndarray,
    mask: jnp.ndarray,
    y: jnp.ndarray,
    tolerance: float = 1e-3,
    max_iters: int = 100,
    near_field_damping: float = 0.9,
    far_field_damping: float = 0.9,
) -> _HIOState:
    """Perform the Damped Error Reduction (ER) algorithm."""
    damped_ER_step_func = jax.tree_util.Partial(
        _damped_ER_projection_step,
        mask=mask,
        y=y,
        near_field_damping=near_field_damping,
        far_field_damping=far_field_damping,
    )
    damped_ER_cond_func = jax.tree_util.Partial(
        _HIO_cond_func, tolerance=tolerance, max_iters=max_iters
    )
    initial_state = _HIOState(
        x=x0,
        iteration=0,
        residual=jnp.linalg.norm(jnp.square(jnp.abs(jnp.fft.fft2(mask * x0))) - y),
    )
    final_state = lax.while_loop(
        damped_ER_cond_func, damped_ER_step_func, initial_state
    )

    return final_state
