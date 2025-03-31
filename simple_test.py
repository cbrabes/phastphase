import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax import random
import jax.lax as lax
import jax.scipy.optimize

from phastphase.retrieval_jax import retrieve

from phastphase.intensity_interpolation import interpolate_intensities


key = random.key(42)
key, subkey = random.split(key)

x_real = random.normal(key, shape=(256,256))
x_complex = random.normal(key, shape=(256,256))

x = jnp.copy(lax.complex(x_real, x_complex))
x = x.at[1,1].set(2*256**2)
y = jnp.square(jnp.abs(jnp.fft.fft2(x,s=(512,512),norm='ortho')))
shape = (256,256)

x_out = retrieve(y, shape,max_iters=1000, grad_tolerance=1e-12, winding_guess=(1,1))


print(jnp.linalg.vector_norm(x_out-x)/jnp.linalg.vector_norm(x))