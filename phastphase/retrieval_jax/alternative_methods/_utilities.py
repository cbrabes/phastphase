import jax.numpy as jnp
from jax import lax


def view_as_flat_real(x):
    r = lax.expand_dims(jnp.real(x), [2])

    c = lax.expand_dims(jnp.imag(x), [2])
    real_out = lax.concatenate((r, c), 2)
    return jnp.ravel(real_out)


def view_as_complex(x, shape):
    x_c = jnp.reshape(x, (shape[0], shape[1], 2))
    return lax.complex(x_c[:, :, 0], x_c[:, :, 1])


def far_field_amplitude_projection(
    far_field: jnp.ndarray, target_amplitudes: jnp.ndarray
) -> jnp.ndarray:
    """
    Perform the far field amplitude projection.

    Args:
        far_field (jnp.ndarray): The far field data.
        target_amplitudes (jnp.ndarray): The target amplitudes.

    Returns:
        jnp.ndarray: The projected far field data.
    """
    return jnp.sign(far_field) * target_amplitudes


def support_projection(
    far_field: jnp.ndarray, support_mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Perform the support projection.

    Args:
        far_field (jnp.ndarray): The far field data.
        support_mask (jnp.ndarray): The near-field support mask.

    Returns:
        jnp.ndarray: The projected far field data.
    """
    return jnp.fft.fft2(
        jnp.fft.ifft2(far_field, norm="ortho") * support_mask, norm="ortho"
    )
