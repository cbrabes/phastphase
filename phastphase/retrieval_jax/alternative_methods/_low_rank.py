import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def low_rank_cost(
    x: jnp.ndarray, intensities: jnp.ndarray, trace_weight: float
) -> ArrayLike:
    trace = jnp.square(jnp.linalg.vector_norm(x))
    far_field_intensities = jnp.square(
        jnp.abs(jnp.fft.fft2(x, norm="ortho", s=intensities.shape))
    )
    total_far_field_intensity = jnp.sum(far_field_intensities, axis=0)
    return (
        jnp.square(jnp.linalg.vector_norm(total_far_field_intensity - intensities))
        + trace_weight * trace
    )
