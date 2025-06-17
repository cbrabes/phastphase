import jax.numpy as jnp


def intensity_loss(
    x: jnp.ndarray,
    target_intensities: jnp.ndarray,
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
    observed_intensities = jnp.square(
        jnp.abs(jnp.fft.fft2(x, s=target_intensities.shape, norm="ortho"))
    )
    return 0.5 * jnp.square(
        jnp.linalg.vector_norm(observed_intensities - target_intensities)
    )


def amplitude_loss(
    x: jnp.ndarray,
    target_amplitudes: jnp.ndarray,
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
    observed_amplitudes = jnp.abs(
        jnp.fft.fft2(x, s=target_amplitudes.shape, norm="ortho")
    )
    return 0.5 * jnp.square(
        jnp.linalg.vector_norm(observed_amplitudes - target_amplitudes)
    )


def weighted_intensity_loss(
    x: jnp.ndarray,
    target_intensities: jnp.ndarray,
    weighting_vector: jnp.ndarray,
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
    observed_intensities = jnp.square(
        jnp.abs(jnp.fft.fft2(x, s=target_intensities.shape, norm="ortho"))
    )
    return 0.5 * jnp.square(
        jnp.linalg.vector_norm(
            weighting_vector * (observed_intensities - target_intensities)
        )
    )


def weighted_amplitude_loss(
    x: jnp.ndarray,
    target_amplitudes: jnp.ndarray,
    weighting_vector: jnp.ndarray,
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
    observed_amplitudes = jnp.abs(
        jnp.fft.fft2(x, s=target_amplitudes.shape, norm="ortho")
    )
    return 0.5 * jnp.square(
        jnp.linalg.vector_norm(
            weighting_vector * (observed_amplitudes - target_amplitudes)
        )
    )
