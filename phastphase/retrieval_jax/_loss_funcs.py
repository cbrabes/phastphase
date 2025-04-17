import jax.numpy as jnp


def normalized_intensity_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(jnp.square(jnp.abs(jnp.fft.fft2(x))) - y) / jnp.linalg.norm(
        y
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
