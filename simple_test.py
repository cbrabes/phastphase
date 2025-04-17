import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.optimize
from jax import random
from phastphase.retrieval_jax import retrieve

jax.config.update("jax_enable_x64", True)


key = random.key(42)
key, subkey = random.split(key)

x_real = random.normal(key, shape=(256, 256))
x_complex = random.normal(key, shape=(256, 256))

x = jnp.copy(lax.complex(x_real, x_complex))
x = x.at[1, 1].set(10 * 256)
y = jnp.square(jnp.abs(jnp.fft.fft2(x, s=(512, 512), norm="ortho")))
shape = (256, 256)
mask = jnp.ones(shape, dtype=jnp.float64)  # Full mask, no masking
x_out, val = retrieve(
    y, mask, max_iters=1000, grad_tolerance=1e-20, winding_guess=(1, 1)
)


print(jnp.linalg.vector_norm(x_out - x) / jnp.linalg.vector_norm(x))
