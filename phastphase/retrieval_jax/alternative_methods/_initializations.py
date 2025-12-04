from typing import Tuple

import jax
import jax.numpy as jnp
from jax import experimental
from jax.experimental.sparse.linalg import lobpcg_standard
from jax.tree_util import Partial

from ._utilities import view_as_complex, view_as_flat_real


def observation_operator(
    x: jnp.ndarray, measured_intensity: jnp.ndarray, near_field_shape: Tuple[int, ...]
) -> jnp.ndarray:
    """Observation operator for phase retrieval.

    Args:
        x: vector being acted on by operator.
        measured_intensity: Measured intensity in the far field.

    Returns:
        action of observation operator on vector x.
    """
    x_c = view_as_complex(x[:, 0], (near_field_shape[0], near_field_shape[1]))
    far_field = jnp.fft.fft2(x_c, s=measured_intensity.shape, norm="ortho")
    near_field = jnp.fft.ifft2(far_field * measured_intensity, norm="ortho")
    near_field = near_field[0 : near_field_shape[0], 0 : near_field_shape[1]]
    return jnp.expand_dims(view_as_flat_real(near_field), 1)


def spectral_initialization(
    near_field_shape: Tuple[int, ...],
    measured_intensity: jnp.ndarray,
    random_key: jnp.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    num_vectors: int = 1,
) -> jnp.ndarray:
    """Spectral initialization for phase retrieval.

    Args:
        near_field_shape: Shape of the near field (height, width).
        measured_intensity: Measured intensity in the far field.

    Returns:
        Initial guess for the near field, derived from Wirtinger flow spectral initialization.
    """
    observation_op = Partial(
        observation_operator,
        measured_intensity=measured_intensity,
        near_field_shape=near_field_shape,
    )

    x = jnp.ones(near_field_shape)
    x = view_as_flat_real(x)
    current_key, subkey = jax.random.split(random_key)
    x_init = jax.random.normal(subkey, (x.size, num_vectors))
    theta, U, i = lobpcg_standard(A=observation_op, X=x_init, tol=tol, m=max_iter)

    sorted_indices = jnp.argsort(-theta)
    principal_eigenvector = U[:, sorted_indices[0]]
    output_x = view_as_complex(principal_eigenvector, near_field_shape)
    output_x = (
        output_x
        / jnp.linalg.vector_norm(output_x)
        * jnp.sqrt(jnp.sum(measured_intensity))
    )
    return output_x


def truncated_spectral_initialization(
    near_field_shape: Tuple[int, ...],
    measured_intensity: jnp.ndarray,
    random_key: jnp.ndarray,
    truncation_threshold: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-6,
    num_vectors: int = 1,
) -> jnp.ndarray:
    """Truncated spectral initialization for phase retrieval.

    Args:
        near_field_shape: Shape of the near field (height, width).
        measured_intensity: Measured intensity in the far field.
        truncation_threshold: Threshold for truncating high-intensity measurements.

    Returns:
        Initial guess for the near field, derived from truncated Wirtinger flow spectral initialization.
    """
    intensity_mean = jnp.mean(measured_intensity)
    threshold_value = truncation_threshold * intensity_mean
    truncated_intensity = jnp.where(
        measured_intensity <= threshold_value, measured_intensity, 0.0
    )

    observation_op = Partial(
        observation_operator,
        measured_intensity=truncated_intensity,
        near_field_shape=near_field_shape,
    )

    x = jnp.ones(near_field_shape)
    x = view_as_flat_real(x)
    current_key, subkey = jax.random.split(random_key)
    x_init = jax.random.normal(subkey, (x.size, num_vectors))
    theta, U, i = lobpcg_standard(A=observation_op, X=x_init, tol=tol, m=max_iter)

    sorted_indices = jnp.argsort(-theta)
    principal_eigenvector = U[:, sorted_indices[0]]
    output_x = view_as_complex(principal_eigenvector, near_field_shape)
    output_x = (
        output_x
        / jnp.linalg.vector_norm(output_x)
        * jnp.sqrt(jnp.sum(measured_intensity))
    )
    return output_x


def orthogonality_promoting_initialization(
    near_field_shape: Tuple[int, ...],
    measured_intensity: jnp.ndarray,
    random_key: jnp.ndarray,
    tol: float = 1e-6,
    truncation_ratio: float = 1.0 / 6,
    num_vectors: int = 1,
    max_iter: int = 1000,
) -> jnp.ndarray:
    """Orthogonality-promoting initialization for phase retrieval.
    Args:
        near_field_shape: Shape of the near field (height, width).
        measured_intensity: Measured intensity in the far field.
        num_iterations: Number of gradient flow iterations.
        step_size: Step size for gradient descent."""

    num_samples = jnp.size(measured_intensity)
    truncated_size = jnp.floor(num_samples * truncation_ratio).astype(int)
    sorted_indices = jnp.argsort(measured_intensity, descending=True)
    unraveled_index = jnp.unravel_index(
        sorted_indices[truncated_size - 1], measured_intensity.shape
    )
    threshold = measured_intensity[unraveled_index]
    truncated_intensity = jnp.where(
        measured_intensity <= threshold, measured_intensity, 0.0
    )
    x = jnp.ones(near_field_shape, dtype=jnp.complex64)
    x = view_as_flat_real(x)
    current_key, subkey = jax.random.split(random_key)
    x_init = jax.random.normal(subkey, (x.size, num_vectors))
    obs_op = Partial(
        observation_operator,
        measured_intensity=truncated_intensity,
        near_field_shape=near_field_shape,
    )
    theta, U, i = lobpcg_standard(A=obs_op, X=x_init, tol=tol, m=max_iter)

    sorted_indices = jnp.argsort(-theta)
    principal_eigenvector = U[:, sorted_indices[0]]
    output_x = view_as_complex(principal_eigenvector, near_field_shape)
    output_x = (
        output_x
        / jnp.linalg.vector_norm(output_x)
        * jnp.sqrt(jnp.sum(measured_intensity))
    )
    return output_x


def random_initialization(
    near_field_shape: Tuple[int, ...],
    measured_intensity: jnp.ndarray,
    random_key: jnp.ndarray,
) -> jnp.ndarray:
    """Random initialization for phase retrieval.
    TODO: Should we wrap the phase?

    Args:
        near_field_shape: Shape of the near field (height, width).
        measured_intensity: Measured intensity in the far field.
        random_seed: Random seed for reproducibility.

    Returns:
        Initial guess for the near field, a random complex array normalized by the square root of the sum of measured intensity.
    """
    current_key, subkey = jax.random.split(random_key)
    x = jax.random.normal(subkey, near_field_shape, dtype=jnp.complex64)
    x_flat = view_as_flat_real(x)
    output_x = view_as_complex(x_flat, near_field_shape)
    output_x = (
        output_x
        / jnp.linalg.vector_norm(output_x)
        * jnp.sqrt(jnp.sum(measured_intensity))
    )
    return output_x
