import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import scipy
import scipy.linalg
from jax import random
import jax.lax as lax
jax.config.update('jax_enable_x64', True)



#@partial(jnp.vectorize, excluded={0}, signature='()->(n)')
def ifft_vec(support, index):
    unit_vector = jnp.zeros_like(support)
    unit_vector = unit_vector.at[index].set(1)
    return jnp.ravel(jnp.fft.ifft2(unit_vector))

def interpolation_operator(support_mask, missing_intensity_indices, autocorr_indices):
    W = jnp.zeros((jnp.size(support_mask), len(missing_intensity_indices[0])), dtype=jnp.complex128)
    missing_index_list = [(missing_intensity_indices[0][i], missing_intensity_indices[1][i]) for i in range(len(missing_intensity_indices[0]))]
    for col_index, intensity_index in enumerate(missing_index_list):
        W = W.at[:, col_index].set(ifft_vec(support_mask, intensity_index))
    P = W[autocorr_indices, :]
    return P

def calculate_residual(intensities, autocorr_indices):
    autocorr = jnp.fft.ifft2(intensities)
    return autocorr[autocorr_indices]

def calc_flat_autocorr_indices(support_mask):
    intensity = jnp.square(jnp.abs(jnp.fft.fft2(support_mask)))
    autocorr = jnp.fft.ifft2(intensity)
    mask = jnp.ones_like(autocorr)
    mask = mask.at[autocorr >0].set(0)
    return jnp.flatnonzero(mask)

def calc_autocorr_indices(support_mask):
    intensity = jnp.square(jnp.abs(jnp.fft.fft2(support_mask)))
    autocorr = jnp.fft.ifft2(intensity)
    mask = jnp.ones_like(autocorr)
    mask = mask.at[autocorr >0].set(0)
    return jnp.nonzero(mask)


def interpolate_intensities(far_field_intensities, #Total far-field pattern, with missing elements set to 0
                            near_field_support_mask, #Support mask, the total size should be the size of the far-field
                            missing_intensity_indices #indices of the missing intensity measurements. Should of the form you get from jax.numpy.nonzero
                            ):
    autocorr_indices = calc_autocorr_indices(near_field_support_mask)
    b_complex = -calculate_residual(far_field_intensities, autocorr_indices)
    interpolation_matrix_complex = interpolation_operator(near_field_support_mask, missing_intensity_indices, calc_flat_autocorr_indices(near_field_support_mask))
    b_real_imag = jnp.concat([jnp.real(b_complex), jnp.imag(b_complex)])
    A_matrix = jnp.concat([jnp.real(interpolation_matrix_complex), jnp.imag(interpolation_matrix_complex)])
    b_numpy = np.asarray(b_real_imag)
    A_numpy = np.asarray(A_matrix)
    x, res, rank, svds = scipy.linalg.lstsq(A_numpy, b_numpy)
    return jnp.asarray(x)


def replace_negative_intensities(far_field_intensities,
                                near_field_support_mask):
    index_mask = jnp.zeros_like(far_field_intensities)
    index_mask = index_mask.at[jnp.less(far_field_intensities,0)].set(1)
    missing_indices = jnp.nonzero(index_mask)
    missing_intensities = interpolate_intensities(far_field_intensities, near_field_support_mask, missing_indices)

    far_field_intensities = far_field_intensities.at[missing_indices].set(missing_intensities)
    return far_field_intensities

