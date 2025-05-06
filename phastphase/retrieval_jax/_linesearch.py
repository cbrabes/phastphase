from typing import Tuple

import jax.numpy as jnp
from jax import Array, lax
from jax.tree_util import Partial
from jax.typing import ArrayLike

_dot = Partial(jnp.dot, precision=lax.Precision.HIGHEST)


def weighted_dot(x: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return _dot(jnp.ravel(x * w), jnp.ravel(y * w))


def polynomial_base_loss_coefficients(
    x: jnp.ndarray,
    scale_factor: ArrayLike,
    search_direction: jnp.ndarray,
    weighting_vector: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes coefficients of polynomial representing loss function along search direction. Loss functions is:
    scale_factor* || w *( |F x|^2 - y)||^2
    """

    search_ff = jnp.fft.fft2(search_direction, s=y.shape, norm="ortho")
    search_intensities = jnp.square(jnp.abs(search_ff))
    x_ff = jnp.fft.fft2(x, s=y.shape, norm="ortho")
    x_intensities = jnp.square(jnp.abs(x_ff))
    partial_dot = jnp.real(search_ff * jnp.conj(x_ff))
    coefficients = jnp.zeros(5)
    c4 = jnp.square(jnp.linalg.vector_norm(search_intensities * weighting_vector))
    c3 = 4 * weighted_dot(search_intensities, partial_dot, weighting_vector)
    c2 = 2 * weighted_dot(
        search_intensities, x_intensities - y, weighting_vector
    ) + 4 * weighted_dot(partial_dot, partial_dot, weighting_vector)
    c1 = 4 * weighted_dot(x_intensities - y, partial_dot, weighting_vector)
    c0 = jnp.square(jnp.linalg.vector_norm(weighting_vector * (x_intensities - y)))
    coefficients = jnp.array([c4, c3, c2, c1, c0])
    return coefficients * scale_factor


def polynomial_phase_loss_coefficients(
    x: jnp.ndarray,
    search_direction: jnp.ndarray,
    phase_ref_point: Tuple[int, int],
    phase_weight: ArrayLike,
) -> jnp.ndarray:
    """
    computes coefficients of polynomial representing loss function along search direction. Loss functions is:
    phase_weight*imag(x[phase_ref_point])^2
    """

    coefficients = jnp.zeros(3)
    c2 = jnp.square(jnp.imag(search_direction[phase_ref_point]))
    c1 = 2 * jnp.imag(search_direction[phase_ref_point]) * jnp.imag(x[phase_ref_point])
    c0 = jnp.square(jnp.imag(x[phase_ref_point]))
    coefficients = jnp.array([c2, c1, c0])
    return coefficients * phase_weight


def _exact_quartic_minimizer(coefficients: jnp.ndarray) -> Array:
    cubic_coeffs = jnp.polyder(coefficients)
    roots = jnp.roots(cubic_coeffs)
    roots = jnp.real(roots)
    function_values = jnp.polyval(coefficients, roots)
    min_index = jnp.argmin(function_values)
    return roots[min_index]


def _closest_quartic_local_minimum(coefficients: jnp.ndarray) -> Array:
    """
    Finds the closest positive critical point a quartic polynomial.
    """

    cubic_coeffs = jnp.polyder(coefficients)
    roots = jnp.roots(cubic_coeffs)
    real_roots = jnp.real(roots)
    root_distances = jnp.abs(roots)

    # Only consider positive roots
    root_distances = jnp.where(
        jnp.greater_equal(real_roots, 0), root_distances, jnp.inf
    )

    # Only consider real roots
    root_distances = jnp.where(
        jnp.less_equal(jnp.abs(jnp.imag(roots)), 1e-10), root_distances, jnp.inf
    )
    min_index = jnp.argmin(root_distances)

    return real_roots[min_index]


def exact_linesearch_intensity_loss(
    x: jnp.ndarray,
    scale_factor: ArrayLike,
    search_direction: jnp.ndarray,
    weighting_vector: jnp.ndarray,
    y: jnp.ndarray,
    phase_ref_point: Tuple[int, int],
    phase_weight: ArrayLike,
    first_minimum: bool = False,
) -> Array:
    """
    Computes exact linesearch for intensity loss function.
    """

    coefficients = polynomial_base_loss_coefficients(
        x, scale_factor, search_direction, weighting_vector, y
    )
    phase_coefficients = polynomial_phase_loss_coefficients(
        x, search_direction, phase_ref_point, phase_weight
    )
    coefficients = coefficients.at[3:].add(phase_coefficients)
    root = lax.cond(
        first_minimum,
        _closest_quartic_local_minimum,
        _exact_quartic_minimizer,
        coefficients,
    )
    return root
