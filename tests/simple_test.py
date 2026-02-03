import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.optimize
from jax import random
from phastphase.retrieval_jax import retrieve

jax.config.update("jax_enable_x64", True)


key = random.key(42)
key, subkey = random.split(key)

x_real = random.normal(key, shape=(32, 64))
x_complex = random.normal(key, shape=(32, 64))

x = jnp.copy(lax.complex(x_real, x_complex))
x = x.at[3, 3].set(10 * 64)
y = jnp.square(jnp.abs(jnp.fft.fft2(x, s=(256, 512), norm="ortho")))
shape = (32, 64)
mask = jnp.ones(shape, dtype=jnp.float64)  # Full mask, no masking
x_out, val = retrieve(
    y,
    mask,
    max_iters=1000,
    descent_method=0,
    grad_tolerance=1e-14,
    winding_guess=(3, 3),
)


print(jnp.linalg.vector_norm(x_out - x) / jnp.linalg.vector_norm(x))
