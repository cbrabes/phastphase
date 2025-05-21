from typing import Callable

import jax.numpy as jnp
from jax import Array, lax
from jax.tree_util import Partial


def view_quotient_as_flat_real(x, winding_index):
    """
    Given a 2D complex array, return a flattened view on the quotient manifold,
    defined as setting the global phase so that the element at winding_index is real and positive.
    """
    r = lax.expand_dims(jnp.real(x), [2])

    c = lax.expand_dims(jnp.imag(x), [2])
    real_out = lax.concatenate((r, c), 2)
    deletion_index = jnp.ravel_multi_index(
        (winding_index[0], winding_index[1], 1), real_out.shape, mode="wrap"
    )
    flat_real_out = jnp.ravel(real_out)
    flat_real_out = jnp.delete(
        flat_real_out, deletion_index, assume_unique_indices=True
    )
    return flat_real_out


def view_quotient_as_complex(x, shape, insertion_index):
    """
    Given real flat view from the quotient maniofold, return the lift on the total space
    """
    x = jnp.insert(x, insertion_index, 0.0)
    x_c = jnp.reshape(x, (shape[0], shape[1], 2))
    return lax.complex(x_c[:, :, 0], x_c[:, :, 1])


def create_quotient_lift_func(
    winding_index: tuple[int, int], shape: tuple[int, int]
) -> Callable:
    """
    Create a function that lifts a point on the quotient manifold to the total space.
    The lifting is done by setting the global phase so that the element at winding_index is real and positive.
    """

    insertion_index = jnp.ravel_multi_index(
        (winding_index[0], winding_index[1], 1), (shape[0], shape[1], 2), mode="wrap"
    )
    quotient_lift_func = Partial(
        view_quotient_as_complex,
        shape=shape,
        insertion_index=insertion_index,
    )
    return quotient_lift_func


def weighted_intensity_loss(
    x_quotient_rep: jnp.ndarray,
    far_field_intensities: jnp.ndarray,
    quotient_lift_func: Callable,
    support_mask: jnp.ndarray,
    weighting_vector: jnp.ndarray,
) -> Array:
    """
    Computes the weighted intensity loss function for a given point on the quotient manifold.
    """
    # We need to lift the point on the quotient manifold to the total space
    x = quotient_lift_func(x_quotient_rep)

    # Apply the support mask
    x = x * support_mask

    # Generate the intensities, using orthonormal fft to improve conditioning of the problem
    guess_intensities = jnp.square(
        jnp.abs(jnp.fft.fft2(x, s=far_field_intensities.shape, norm="ortho"))
    )

    # We divide by 8 to keep the eigenvalues clustered around 1
    return (
        jnp.square(
            jnp.linalg.vector_norm(
                weighting_vector * (guess_intensities - far_field_intensities)
            )
        )
        / 8
    )
