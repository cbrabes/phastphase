from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from ._loss_functions import (
    amplitude_loss,
    weighted_amplitude_loss,
    weighted_intensity_loss,
)


class GradientFlowState(NamedTuple):
    x_k: jnp.ndarray
    gradient_norm_k: jnp.ndarray
    step_number: int


def _truncated_gradient_flow_step(
    state: GradientFlowState,
    gradient_function: Callable,
    y: jnp.ndarray,
    weighting_vector: jnp.ndarray,
    truncation_threshold: float = 0.7,
    step_length: float = 1.0,
    decay_constant: float = 300,
):
    far_field_abs = jnp.abs(jnp.fft.fft2(state.x_k, s=y.shape, norm="ortho"))
    mask = far_field_abs > (1 / (1 + truncation_threshold)) * y
    gradient = gradient_function(state.x_k, weighting_vector=mask * weighting_vector)

    used_step_length = jnp.fmin(
        1 - jnp.exp(-state.step_number / decay_constant), step_length
    )
    new_x = state.x_k - used_step_length * jnp.conj(gradient)
    return GradientFlowState(
        x_k=new_x,
        gradient_norm_k=jnp.linalg.norm(
            gradient_function(state.x_k, weighting_vector=weighting_vector)
        ),
        step_number=state.step_number + 1,
    )


def _gradient_flow_step(
    state: GradientFlowState,
    gradient_function: Callable,
    max_step_length: float = 1.0,
    decay_constant: float = 300,
):
    gradient = gradient_function(state.x_k)
    step_length = jnp.fmin(
        1 - jnp.exp(-state.step_number / decay_constant), max_step_length
    )
    new_x = state.x_k - step_length * jnp.conj(gradient)
    return GradientFlowState(
        x_k=new_x,
        gradient_norm_k=jnp.linalg.norm(gradient),
        step_number=state.step_number + 1,
    )


def _gradient_flow_cond(
    state: GradientFlowState,
    grad_tolerance: float = 1e-3,
    iter_limit: int = 1000,
):
    return jnp.logical_and(
        jnp.greater(state.gradient_norm_k, grad_tolerance),
        jnp.less(state.step_number, iter_limit),
    )


def truncated_flow_loop(
    x_0,
    y,
    grad_function,
    weighting_vector: jnp.ndarray,
    truncation_threshold=0.7,
    step_length=1.0,
    grad_tolerance=1e-3,
    iter_limit=1000,
    decay_constant=300,
):
    body_func = Partial(
        _truncated_gradient_flow_step,
        gradient_function=grad_function,
        y=y,
        truncation_threshold=truncation_threshold,
        step_length=step_length,
        weighting_vector=weighting_vector,
        decay_constant=decay_constant,
    )
    cond_func = Partial(
        _gradient_flow_cond, grad_tolerance=grad_tolerance, iter_limit=iter_limit
    )
    final_state = jax.lax.while_loop(
        cond_func,
        body_func,
        GradientFlowState(
            x_k=x_0,
            gradient_norm_k=jnp.linalg.norm(grad_function(x_0, weighting_vector=1 / y)),
            step_number=0,
        ),
    )
    return final_state.x_k


def flow_loop(
    x_0,
    grad_function,
    max_step_length=1.0,
    grad_tolerance=1e-3,
    iter_limit=1000,
    decay_constant=300,
):
    body_func = Partial(
        _gradient_flow_step,
        gradient_function=grad_function,
        max_step_length=max_step_length,
        decay_constant=decay_constant,
    )
    cond_func = Partial(
        _gradient_flow_cond, grad_tolerance=grad_tolerance, iter_limit=iter_limit
    )
    final_state = jax.lax.while_loop(
        cond_func,
        body_func,
        GradientFlowState(
            x_k=x_0,
            gradient_norm_k=jnp.linalg.norm(grad_function(x_0)),
            step_number=0,
        ),
    )
    return final_state.x_k


def amplitude_flow(
    x_0,
    far_field_intensities,
    step_length=1.0,
    grad_tolerance=1e-3,
    iter_limit=1000,
    decay_constant=300,
):
    y = jnp.sqrt(far_field_intensities)
    loss_func = Partial(amplitude_loss, target_amplitudes=y)
    grad_func = jax.grad(loss_func, argnums=0)
    return flow_loop(
        x_0,
        grad_func,
        max_step_length=step_length,
        grad_tolerance=grad_tolerance,
        iter_limit=iter_limit,
        decay_constant=decay_constant,
    )


def wirtinger_flow(
    x_0,
    far_field_intensities,
    step_length=1.0,
    grad_tolerance=1e-3,
    iter_limit=1000,
    decay_constant=300,
):
    loss_func = Partial(
        weighted_intensity_loss,
        target_intensities=far_field_intensities,
        weighting_vector=1 / jnp.sqrt(far_field_intensities),
    )
    grad_func = jax.grad(loss_func, argnums=0)
    return flow_loop(
        x_0,
        grad_func,
        max_step_length=step_length,
        grad_tolerance=grad_tolerance,
        iter_limit=iter_limit,
        decay_constant=decay_constant,
    )


def truncated_amplitude_flow(
    x_0,
    far_field_intensities,
    truncation_threshold=0.7,
    step_length=1.0,
    grad_tolerance=1e-3,
    iter_limit=1000,
    decay_constant=300,
):
    loss_func = Partial(
        weighted_amplitude_loss, target_amplitudes=jnp.sqrt(far_field_intensities)
    )
    grad_func = jax.grad(loss_func, argnums=0)
    return truncated_flow_loop(
        x_0,
        jnp.sqrt(far_field_intensities),
        grad_func,
        truncation_threshold=truncation_threshold,
        step_length=step_length,
        grad_tolerance=grad_tolerance,
        iter_limit=iter_limit,
        weighting_vector=1,
        decay_constant=decay_constant,
    )


def truncated_wirtinger_flow(
    x_0,
    far_field_intensities,
    truncation_threshold=0.7,
    step_length=1.0,
    grad_tolerance=1e-3,
    iter_limit=1000,
    decay_constant=300,
):
    loss_func = Partial(
        weighted_intensity_loss, target_intensities=far_field_intensities
    )
    grad_func = jax.grad(loss_func, argnums=0)
    return truncated_flow_loop(
        x_0,
        jnp.sqrt(far_field_intensities),
        grad_func,
        truncation_threshold=truncation_threshold,
        step_length=step_length,
        grad_tolerance=grad_tolerance,
        iter_limit=iter_limit,
        weighting_vector=1 / jnp.sqrt(far_field_intensities),
        decay_constant=decay_constant,
    )
