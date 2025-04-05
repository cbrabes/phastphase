import jax
import jax.numpy as jnp



def calculate_autocorr_support(near_field_support_mask):

    ff_intensity = jnp.abs(jnp.square(jnp.fft.fft2(near_field_support_mask)))

    return jnp.fft.ifft2(ff_intensity)


def intensity_support_projection(far_field_intensities,
                                 near_field_suppport_mask):
    autocorr_support = calculate_autocorr_support(near_field_suppport_mask)
    ff_autocorr = jnp.fft.ifft2(far_field_intensities)
    ff_autocorr = ff_autocorr.at[jnp.less_equal(autocorr_support,0)].set(0)
    projected_intensities = jnp.fft.fft2(ff_autocorr)
    return projected_intensities