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


def normalized_intensity_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(jnp.square(jnp.abs(jnp.fft.fft2(x))) - y) / jnp.linalg.norm(
        y
    )


def sqrt_intensity_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(
        (jnp.square(jnp.abs(jnp.fft.fft2(x))) - y) / jnp.sqrt(y + 1e-12)
    )


def intensity_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute the intensity loss between the Fourier transform of x and y.

    Args:
        x (jnp.ndarray): The input array.
        y (jnp.ndarray): The target array.

    Returns:
        jnp.ndarray: The intensity loss value.
    """
    return jnp.linalg.norm(jnp.square(jnp.abs(jnp.fft.fft2(x))) - y)


def intensity_residuals(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute the residuals for intensity loss.

    Args:
        x (jnp.ndarray): The input array.
        y (jnp.ndarray): The target array.

    Returns:
        jnp.ndarray: The residuals, which are the differences between the squared magnitudes of the Fourier transforms.
    """
    return jnp.square(jnp.abs(jnp.fft.fft2(x, s=y.shape, norm="ortho"))) - y


def weighted_intensity_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    weights: jnp.ndarray,
    x_shape: tuple,
    scale_factor: float = 1.0 / 8,
    phase_weight: float = 1.0,
    phase_ref_point: tuple = (0, 0),
) -> jnp.ndarray:
    """Compute the weighted intensity loss between the Fourier transform of x and y.

    Args:
        x (jnp.ndarray): The input array.
        y (jnp.ndarray): The target array.
        weights (jnp.ndarray): The weights for the loss calculation.
        phase_weight (float, optional): The weight for the phase regularization term. Defaults to 1.0.
        phase_ref_point (tuple, optional): The reference point for phase regularization. Defaults to (0, 0).

    Returns:
        jnp.ndarray: The weighted intensity loss value.
    """
    x_c = view_as_complex(x, x_shape)
    return scale_factor * jnp.square(
        jnp.linalg.vector_norm(
            (jnp.square(jnp.abs(jnp.fft.fft2(x_c, s=y.shape, norm="ortho"))) - y)
            * weights
        )
    ) + phase_weight * jnp.square(jnp.imag(x_c[*phase_ref_point]))
