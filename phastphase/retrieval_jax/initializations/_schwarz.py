from typing import Tuple

import jax.numpy as jnp
import scipy
from scipy.sparse.linalg import LinearOperator, lsmr


def find_autocorr_mask(
    unknown_mask: jnp.ndarray, known_mask: jnp.ndarray
) -> jnp.ndarray:
    return None


# Here's what we are going to do:
# Define function that performs autocorrelation, then define jvp and vjp for it.


def autocorr(
    x: jnp.ndarray,
    v: jnp.ndarray,
    shape: Tuple[int, int],
    far_field_shape: Tuple[int, int],
) -> jnp.ndarray:
    """
    Perform autocorrelation on a 2D array.
    """
    return jnp.fft.ifft2(x, s=shape, norm="ortho")

