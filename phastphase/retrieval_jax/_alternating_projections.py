from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ._loss_funcs import normalized_intensity_loss, sqrt_intensity_loss
from .alternative_methods._alternating_minimization import Rs, Rm, Pm


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
    y_c = jnp.fft.fft2(state.x, norm="ortho")
    y_c = jnp.sign(y_c) * jnp.sqrt(y)
    x_new = jnp.fft.ifft2(y_c, norm="ortho")
    x_new = mask * (x_new) + (1 - mask) * (state.x - beta * x_new)
    residual = sqrt_intensity_loss(mask * x_new, y)
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
    **kwargs
) -> _HIOState:
    """
    Runs the Hybrid Input-Output (HIO) algorithm given an initial seed object `x0`.

    The HIO algorithm is an iterative phase retrieval method used to reconstruct an object
    from its Fourier Intensity measurements. It alternates between enforcing constraints
    in the object domain and the Fourier domain, with a feedback parameter `beta` to
    control the update step.

        x0 (jnp.ndarray): The initial guess for the object. This should be a 2D array
            representing the spatial domain of the object.
        mask (jnp.ndarray): The near-field support mask, which defines the region of
            interest in the object domain. It should have the same shape as `y`.
        y (jnp.ndarray): The target array representing the Fourier Intensity measurements.
        tolerance (float, optional): The tolerance for convergence. The algorithm stops
            when the the residual (norm of the difference between the current and target Fourier magnitudes)
            is below this value. Defaults to 1e-3.
        max_iters (int, optional): The maximum number of iterations to run the algorithm.
            Defaults to 100.
        beta (float, optional): The feedback parameter used in the HIO update step.
            Controls the influence of the previous estimate on the current update.
            Defaults to 0.9.

    Returns:
        _HIOState: A dataclass containing the final state of the algorithm, including:
            - `x` (jnp.ndarray): The reconstructed object.
            - `iteration` (int): The number of iterations performed.
            - `residual` (float): The final residual value indicating the difference
              between the current and target Fourier magnitudes.

    """
    HIO_step_func = jax.tree_util.Partial(
        _HIO_projection_step, mask=mask, y=y, beta=beta
    )
    HIO_cond_func = jax.tree_util.Partial(
        _HIO_cond_func, tolerance=tolerance, max_iters=max_iters
    )
    initial_state = _HIOState(
        x=x0,
        iteration=0,
        residual=jnp.linalg.norm(
            (jnp.square(jnp.abs(jnp.fft.fft2(mask * x0, norm="ortho"))) - y)
            / jnp.sqrt(y + 1e-12)
        ),
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
    y_c = jnp.fft.fft2(state.x, norm="ortho")
    y_c = far_field_damping * y_c + (1 - far_field_damping) * jnp.sqrt(y) * jnp.sign(
        y_c
    )  # enforce magnitude
    x_new = jnp.fft.ifft2(y_c, norm="ortho")

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
    **kwargs
) -> _HIOState:
    """
    Perform the Damped Error Reduction (ER) algorithm.

    This function implements the Damped Error Reduction (ER) algorithm for phase retrieval.
    It iteratively updates an initial guess `x0` to minimize the residual error between the
    measured data `y` and the Fourier transform of the masked input.

    Args:
        x0 (jnp.ndarray): Initial guess for the solution.
        mask (jnp.ndarray): Binary mask applied to the input in the near field.
        y (jnp.ndarray): Measured intensity data in the far field.
        tolerance (float, optional): Convergence tolerance for the residual error. Defaults to 1e-3.
        max_iters (int, optional): Maximum number of iterations to perform. Defaults to 100.
        near_field_damping (float, optional): Damping factor for the near field updates. Defaults to 0.9.
        far_field_damping (float, optional): Damping factor for the far field updates. Defaults to 0.9.

    Returns:
        _HIOState: A dataclass containing the final state of the algorithm, including:
            - `x` (jnp.ndarray): The reconstructed object.
            - `iteration` (int): The number of iterations performed.
            - `residual` (float): The final residual value indicating the difference
              between the current and target Fourier magnitudes.
    """
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
        residual=jnp.linalg.norm(
            (jnp.square(jnp.abs(jnp.fft.fft2(mask * x0, norm="ortho"))) - y)
            / jnp.sqrt(y + 1e-12)
        ),
    )
    final_state = lax.while_loop(
        damped_ER_cond_func, damped_ER_step_func, initial_state
    )

    return final_state


def ER(
    x0: jnp.ndarray,
    mask: jnp.ndarray,
    y: jnp.ndarray,
    tolerance: float = 1e-3,
    max_iters: int = 100,
    **kwargs
) -> _HIOState:
    """
    Perform the Error Reduction (ER) algorithm.

    This function implements the Error Reduction (ER) algorithm for phase retrieval.
    It iteratively updates an initial guess `x0` to minimize the residual error between the
    measured data `y` and the Fourier transform of the masked input.

    Args:
        x0 (jnp.ndarray): Initial guess for the solution.
        mask (jnp.ndarray): Binary mask applied to the input in the near field.
        y (jnp.ndarray): Measured intensity data in the far field.
        tolerance (float, optional): Convergence tolerance for the residual error. Defaults to 1e-3.
        max_iters (int, optional): Maximum number of iterations to perform. Defaults to 100.

    Returns:
        _HIOState: A dataclass containing the final state of the algorithm, including:
            - `x` (jnp.ndarray): The reconstructed object.
            - `iteration` (int): The number of iterations performed.
            - `residual` (float): The final residual value indicating the difference
              between the current and target Fourier magnitudes.
    """
    return damped_ER(
        x0=x0,
        mask=mask,
        y=y,
        tolerance=tolerance,
        max_iters=max_iters,
        near_field_damping=0.0,
        far_field_damping=0.0,
    )


def _RAAR_projection_step(
    state: _HIOState,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    beta: float = 0.9,
) -> jnp.ndarray:
    """
    Perform the Relaxed Averaged Alternating Reflection (RAAR) step.

    Args:
        state (_HIOState): The current state of the HIO algorithm.
        y (jnp.ndarray): The target amplitudes.
        mask (jnp.ndarray): The near-field support mask.

    Returns:
        jnp.ndarray: The updated estimate of the image.
    """
    y_c = jnp.fft.fft2(state.x, norm="ortho")
    y_c = jnp.sign(y_c) * jnp.sqrt(y)
    x_new = jnp.fft.ifft2(y_c, norm="ortho")
    x_new = mask * (x_new) + (1 - mask) * (state.x - beta * x_new)

    x_new = 1 / 2 * beta * (Rs(Rm(state.x, y), mask) + state.x) + (
        1 - beta
    ) * Pm(state.x, y)

    residual = normalized_intensity_loss(mask * x_new, y)

    return _HIOState(x=x_new, iteration=state.iteration + 1, residual=residual)


def RAAR(
    x0: jnp.ndarray,
    mask: jnp.ndarray,
    y: jnp.ndarray,
    tolerance: float = 1e-3,
    max_iters: int = 100,
    beta: float = 0.9,
    **kwargs
) -> _HIOState:
    """
    Perform the Relaxed Averaged Alternating Reflections (RAAR) algorithm.

    This function implements the Relaxed Averaged Alternating Reflections (RAAR) algorithm for phase retrieval.
    It iteratively updates an initial guess `x0` to minimize the residual error between the
    measured data `y` and the Fourier transform of the masked input.

    Args:
        x0 (jnp.ndarray): Initial guess for the solution.
        mask (jnp.ndarray): Binary mask applied to the input in the near field.
        y (jnp.ndarray): Measured intensity data in the far field.
        tolerance (float, optional): Convergence tolerance for the residual error. Defaults to 1e-3.
        max_iters (int, optional): Maximum number of iterations to perform. Defaults to 100.
        beta (float, optional): Relaxation parameter for the RAAR algorithm. Defaults to 0.9.

    Returns:
        _HIOState: A dataclass containing the final state of the algorithm, including:
            - `x` (jnp.ndarray): The reconstructed object.
            - `iteration` (int): The number of iterations performed.
            - `residual` (float): The final residual value indicating the difference
              between the current and target Fourier magnitudes.
    """
    RAAR_step_func = jax.tree_util.Partial(
        _RAAR_projection_step,
        mask=mask,
        y=y,
        beta=beta,
    )
    RAAR_cond_func = jax.tree_util.Partial(
        _HIO_cond_func, tolerance=tolerance, max_iters=max_iters
    )
    initial_state = _HIOState(
        x=x0,
        iteration=0,
        residual=jnp.linalg.norm(
            (jnp.square(jnp.abs(jnp.fft.fft2(mask * x0, norm="ortho"))) - y)
            / jnp.sqrt(y + 1e-12)
        ),
    )
    final_state = lax.while_loop(
        RAAR_cond_func, RAAR_step_func, initial_state
    )

    return final_state
