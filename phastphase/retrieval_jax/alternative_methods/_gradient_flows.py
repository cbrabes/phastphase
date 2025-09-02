from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import Partial

from ._loss_functions import (
    amplitude_loss,
    weighted_amplitude_loss,
    weighted_intensity_loss,
)
from ._utilities import view_as_complex, view_as_flat_real


class GradientFlowState(NamedTuple):
    params: optax.Params
    opt_state: optax.OptState
    gradient_norm_k: jnp.ndarray
    step_number: int


def _gradient_flow_cond(
    state: GradientFlowState,
    grad_tolerance: float = 1e-3,
    iter_limit: int = 1000,
):
    return jnp.logical_and(
        jnp.greater(state.gradient_norm_k, grad_tolerance),
        jnp.less(state.step_number, iter_limit),
    )


def _gradient_flow_step(
    state: GradientFlowState,
    value_grad_func: Callable,
    loss_func: Callable[[jnp.ndarray], jnp.ndarray],
    solver: optax.GradientTransformationExtraArgs,
) -> GradientFlowState:
    value, grad = value_grad_func(state.params, state=state.opt_state)
    updates, opt_state = solver.update(
        grad, state.opt_state, state.params, value=value, grad=grad, value_fn=loss_func
    )
    params = optax.apply_updates(state.params, updates)
    gradient_norm_k = jnp.linalg.norm(grad)
    step_number = state.step_number + 1
    return GradientFlowState(
        params=params,
        opt_state=opt_state,
        gradient_norm_k=gradient_norm_k,
        step_number=step_number,
    )


def backtracking_gradient_descent(
    x_0: jnp.ndarray,
    loss_func: Callable[[jnp.ndarray], jnp.ndarray],
    grad_tolerance: float = 1e-3,
    iter_limit: int = 1000,
):
    solver = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_zoom_linesearch(max_linesearch_steps=15),
    )
    val_grad_fun = optax.value_and_grad_from_state(loss_func)
    opt_state = solver.init(x_0)
    init_val, init_grad = val_grad_fun(x_0, state=opt_state)
    flow_step = Partial(
        _gradient_flow_step,
        value_grad_func=val_grad_fun,
        loss_func=loss_func,
        solver=solver,
    )
    flow_cond = Partial(
        _gradient_flow_cond,
        grad_tolerance=grad_tolerance,
        iter_limit=iter_limit,
    )
    initial_state = GradientFlowState(
        params=x_0,
        opt_state=opt_state,
        gradient_norm_k=jnp.linalg.vector_norm(init_grad),
        step_number=0,
    )
    final_state = jax.lax.while_loop(
        flow_cond,
        flow_step,
        initial_state,
    )
    return final_state.params


def amplitude_flow(
    x_0,
    far_field_intensities,
    grad_tolerance=1e-3,
    iter_limit=1000,
):
    y = jnp.sqrt(far_field_intensities)
    loss_func = Partial(amplitude_loss, target_amplitudes=y, shape=x_0.shape)
    x_0_real = view_as_flat_real(x_0)
    x_out = backtracking_gradient_descent(
        x_0_real, loss_func, grad_tolerance, iter_limit
    )
    return view_as_complex(x_out, x_0.shape)


def wirtinger_flow(
    x_0,
    far_field_intensities,
    grad_tolerance=1e-3,
    iter_limit=1000,
):
    loss_func = Partial(
        weighted_intensity_loss,
        target_intensities=far_field_intensities,
        weighting_vector=1 / jnp.sqrt(far_field_intensities),
        shape=x_0.shape,
    )
    x_0_real = view_as_flat_real(x_0)
    x_out = backtracking_gradient_descent(
        x_0_real, loss_func, grad_tolerance, iter_limit
    )
    return view_as_complex(x_out, x_0.shape)


def truncated_amplitude_loss(
    x_real: jnp.ndarray,
    target_amplitudes: jnp.ndarray,
    shape,
    truncation_threshold: Optional[float] = 0.7,
) -> jnp.ndarray:
    """
    Computes L2-Norm of loss between target amplitudes and amplitudes of Unitary DFT of x (padded to the Size of y).

    Args:
        x (jnp.ndarray): The predicted amplitudes.
        target_amplitudes (jnp.ndarray): The target amplitudes.
        weighting_vector (jnp.ndarray): The weighting vector.

    Returns:
        jnp.ndarray: The computed intensity loss.
    """
    x = view_as_complex(x_real, shape)
    observed_amplitudes = jnp.abs(
        jnp.fft.fft2(x, s=target_amplitudes.shape, norm="ortho")
    )
    weight = jnp.where(
        observed_amplitudes >= (1 / (1 + truncation_threshold)) * target_amplitudes,
        1.0,
        0.0,
    )
    return 0.5 * jnp.square(
        jnp.linalg.vector_norm(weight * (observed_amplitudes - target_amplitudes))
    )


def truncated_amplitude_flow(
    x_0,
    far_field_intensities,
    truncation_threshold=0.7,
    grad_tolerance=1e-3,
    iter_limit=1000,
):
    loss_func = Partial(
        truncated_amplitude_loss,
        target_amplitudes=jnp.sqrt(far_field_intensities),
        shape=x_0.shape,
        truncation_threshold=truncation_threshold,
    )
    x_0_real = view_as_flat_real(x_0)
    x_out = backtracking_gradient_descent(
        x_0_real, loss_func, grad_tolerance, iter_limit
    )
    return view_as_complex(x_out, x_0.shape)


def truncated_weighted_intensity_loss(
    x_real: jnp.ndarray,
    target_intensities: jnp.ndarray,
    weighting_vector: jnp.ndarray,
    shape,
    truncation_threshold: Optional[float] = 0.7,
) -> jnp.ndarray:
    """
    Computes L2-Norm of loss between target intensities and intensity of Unitary DFT of x (padded to the Size of y).

    Args:
        x (jnp.ndarray): The predicted intensities.
        target_intensities (jnp.ndarray): The target intensities.
        weighting_vector (jnp.ndarray): The weighting vector.

    Returns:
        jnp.ndarray: The computed intensity loss.
    """
    x = view_as_complex(x_real, shape)
    observed_intensities = jnp.square(
        jnp.abs(jnp.fft.fft2(x, s=target_intensities.shape, norm="ortho"))
    )

    truncation_mask = jnp.where(
        observed_intensities >= (1 / (1 + truncation_threshold)) * target_intensities,
        1.0,
        0.0,
    )
    return 0.5 * jnp.square(
        jnp.linalg.vector_norm(
            truncation_mask
            * weighting_vector
            * (observed_intensities - target_intensities)
        )
    )


def truncated_wirtinger_flow(
    x_0,
    far_field_intensities,
    truncation_threshold=0.7,
    grad_tolerance=1e-3,
    iter_limit=1000,
):
    loss_func = Partial(
        truncated_weighted_intensity_loss,
        target_intensities=far_field_intensities,
        weighting_vector=1 / jnp.sqrt(far_field_intensities),
        shape=x_0.shape,
        truncation_threshold=truncation_threshold,
    )
    x_0_real = view_as_flat_real(x_0)
    x_out = backtracking_gradient_descent(
        x_0_real, loss_func, grad_tolerance, iter_limit
    )
    return view_as_complex(x_out, x_0.shape)
