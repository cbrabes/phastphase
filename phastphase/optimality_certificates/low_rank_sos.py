import jax
import jax.numpy as jnp
import jax.lax as lax
from .trust_region import minimize_trust_region
from typing import Callable, NamedTuple, Type, Union, Optional, Tuple

def view_as_flat_real(x):
    r = lax.expand_dims(jnp.real(x),[3])

    c = lax.expand_dims(jnp.imag(x),[3])
    real_out = lax.concatenate((r,c),3)
    return jnp.ravel(real_out)

def view_as_complex(x, shape):
    x_c = jnp.reshape(x, (*shape, 2))
    return lax.complex(x_c[:, :,:,0], x_c[:,:,:,1])

def masked_L2_mag_loss(x,
                y,
                mask,
                shape):
    x_c = mask*view_as_complex(x, shape)

    composite_intensities = jnp.square(jnp.abs(jnp.fft.fft2(jnp.fft.fft2(x_c,s=y.shape,norm='ortho'))))
    sum_intensities = jnp.sum(composite_intensities, axis=0)
    return jnp.square(jnp.linalg.vector_norm(sum_intensities/y-y))/8

def generate_random_guess():
    return None