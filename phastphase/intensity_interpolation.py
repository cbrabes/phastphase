import jax
import jax.numpy as jnp
from functools import partial

@partial(jnp.vectorize, excluded={0}, signature='()->(n)')
def ifft_vec(support, index):
    unit_vector = jnp.zeros_like(support)
    unit_vector = unit_vector.at[index].set(1)
    return jnp.ravel(jnp.fft.ifft2(unit_vector))

def interpolation_operator(support, intensity_indices, autocorr_indices):
    W = ifft_vec(support, intensity_indices)
    P = W[autocorr_indices, :]
    return P

def calculate_residual(intensities, autocorr_indices):
    autocorr = jnp.fft.ifft2(jnp.square(jnp.abs(jnp.fft.fft2(intensities))))
    return autocorr[autocorr_indices]

    
