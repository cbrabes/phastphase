import jax.numpy as jnp

from ._utilities import far_field_amplitude_projection, support_projection

Ps = support_projection
Pm = far_field_amplitude_projection


def Rs(x, support_mask):
    return 2 * Ps(x, support_mask) - x


def Rm(x, target_amplitudes):
    return 2 * Pm(x, target_amplitudes) - x


def error_reduction_step(
    x: jnp.ndarray,
    target_amplitudes: jnp.ndarray,
    support_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Perform the error reduction step.

    Args:
        x (jnp.ndarray): The current estimate of the image.
        target_amplitudes (jnp.ndarray): The target amplitudes.
        support_mask (jnp.ndarray): The near-field support mask.

    Returns:
        jnp.ndarray: The updated estimate of the image.
    """
    return Ps(Pm(x, target_amplitudes), support_mask)


def HIO_step(
    x: jnp.ndarray,
    target_amplitudes: jnp.ndarray,
    support_mask: jnp.ndarray,
    relaxation_factor: float = 0.9,
) -> jnp.ndarray:
    """
    Perform the Hybrid Input-Output (HIO) step.

    Args:
        x (jnp.ndarray): The current estimate of the image.
        target_amplitudes (jnp.ndarray): The target amplitudes.
        support_mask (jnp.ndarray): The near-field support mask.

    Returns:
        jnp.ndarray: The updated estimate of the image.
    """
    return (
        relaxation_factor * Ps(Pm(x, target_amplitudes), support_mask)
        + x
        - Ps(x, support_mask)
        - relaxation_factor * Pm(x, target_amplitudes)
    )


def difference_map_step(
    x: jnp.ndarray,
    target_amplitudes: jnp.ndarray,
    support_mask: jnp.ndarray,
    gamma_s: float = 0.9,
    gamma_m: float = 0.9,
    beta: float = 0.9,
) -> jnp.ndarray:
    """
    Perform the difference map step.

    Args:
        x (jnp.ndarray): The current estimate of the image.
        target_amplitudes (jnp.ndarray): The target amplitudes.
        support_mask (jnp.ndarray): The near-field support mask.

    Returns:
        jnp.ndarray: The updated estimate of the image.
    """
    projection_1 = beta * Ps(
        (1 + gamma_s) * Pm(x, target_amplitudes) - gamma_s * x, support_mask
    )
    projection_2 = beta * Pm(
        (1 + gamma_m) * Ps(x, support_mask) - gamma_m * x, target_amplitudes
    )
    return x + +projection_1 - projection_2


def RAAR_step(
    x: jnp.ndarray,
    target_amplitudes: jnp.ndarray,
    support_mask: jnp.ndarray,
    beta: float = 0.9,
) -> jnp.ndarray:
    """
    Perform the Relaxed Averaged Alternating Reflection (RAAR) step.

    Args:
        x (jnp.ndarray): The current estimate of the image.
        target_amplitudes (jnp.ndarray): The target amplitudes.
        support_mask (jnp.ndarray): The near-field support mask.

    Returns:
        jnp.ndarray: The updated estimate of the image.
    """
    return 1 / 2 * beta * (Rs(Rm(x, target_amplitudes), support_mask) + x) + (
        1 - beta
    ) * Pm(x, target_amplitudes)
