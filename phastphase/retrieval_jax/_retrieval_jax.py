from functools import partial
from typing import Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from ._descent_methods import lbfgs_minimize
from ._trust_region import minimize_trust_region


def view_as_flat_real(x):
    r = lax.expand_dims(jnp.real(x), [2])

    c = lax.expand_dims(jnp.imag(x), [2])
    real_out = lax.concatenate((r, c), 2)
    return jnp.ravel(real_out)


def view_as_complex(x, shape):
    x_c = jnp.reshape(x, (shape[0], shape[1], 2))
    return lax.complex(x_c[:, :, 0], x_c[:, :, 1])


""" 
Constants to define loss functions for jax switch 
"""

L2_MAG_LOSS: int = 0

POISSON_LOSS: int = 1

MAG_LOSS: int = 2

TRUST_REGION_CG: int = 0

LBFGS: int = 1


@jax.jit
def retrieve(
    far_field_intensities: jax.typing.ArrayLike,  # Magnitude squared of fourier transform
    support_mask: jax.typing.ArrayLike,  # Support mask
    winding_guess=(0, 0),  # Guess for the winding number
    offset: float = 1e-14,  # Value to offset y to ensure it is positive
    scale_gradient: bool = False,  # Whether to scale the loss funciton such that the initial infinity norm of the gradient is no more than 1
    max_iters: int = 100,  # Maximum Iterations for use in minimization
    grad_tolerance: float = 1e-5,  # Gradient tolerance w/respect to infinity norm of gradient
    tv_reg: float = 0.0,  # Regularization Parameter for TV regularization
    phase_reg: float = 1.0,
    far_field_mask: Optional[
        jax.typing.ArrayLike
    ] = None,  # Far field mask for cropped far_fields
    loss_type=L2_MAG_LOSS,
    descent_method=LBFGS,
) -> Tuple[jax.Array, float]:  # Loss Type
    winding_number = winding_guess
    mask = support_mask
    support_shape = mask.shape

    if far_field_mask is None:
        ff_mask = jnp.ones_like(far_field_intensities)
    else:
        ff_mask = far_field_mask

    offset_intensities = far_field_intensities + offset
    x_schwarz = fast_schwarz_transform(
        offset_intensities, winding_number, support_shape
    )

    return refine(
        x_schwarz,
        offset_intensities,
        mask,
        winding_number,
        scale_gradient,
        max_iters,
        grad_tolerance,
        tv_reg=tv_reg,
        far_field_mask=ff_mask,
        loss_type=loss_type,
        phase_reg=phase_reg,
        descent_method=descent_method,
    )


def winding_calc(
    far_field_intensities: jnp.ndarray, support_mask: jnp.ndarray
) -> Tuple[jax.typing.ArrayLike, ...]:
    autocorr = jnp.fft.ifft2(far_field_intensities)
    autocorr_mag = jnp.abs(autocorr)
    convolution = jnp.fft.ifft2(
        jnp.fft.fft2(autocorr_mag) * jnp.fft.fft2(support_mask, s=autocorr_mag.shape)
    )
    convolution_mag = jnp.abs(convolution)

    max_loc = jnp.unravel_index(jnp.argmax(convolution_mag), convolution_mag.shape)
    return max_loc


def schwarz_transform(y, winding_tuple, support_shape):
    cepstrum = jnp.fft.ifft2(jnp.log(y))
    cep_mask = jnp.zeros_like(cepstrum)
    fft_freqs_axis0 = jnp.fft.fftfreq(y.shape[0], d=1.0 / y.shape[0])
    fft_freqs_axis1 = jnp.fft.fftfreq(y.shape[1], d=1.0 / y.shape[1])
    x_grid, y_grid = jnp.meshgrid(fft_freqs_axis0, fft_freqs_axis1, indexing="ij")

    cep_mask = cep_mask.at[0 : y.shape[0] // 2, 0 : y.shape[1] // 2].set(1)
    cep_mask = cep_mask.at[
        0 : 2 * winding_tuple[0] + 1, 0 : 2 * winding_tuple[1] + 1
    ].set(0.5)

    rolled_mask = jnp.roll(cep_mask, (-winding_tuple[0], -winding_tuple[1]), (0, 1))
    rolled_mask = rolled_mask.at[
        winding_tuple[1] * x_grid + winding_tuple[0] * y_grid > 0
    ].set(1)
    rolled_mask = rolled_mask.at[
        0 : winding_tuple[0] + 1, 0 : winding_tuple[1] + 1
    ].set(0.5)
    x_cep = jnp.fft.ifft2(jnp.exp(jnp.fft.fft2(rolled_mask * cepstrum)), norm="ortho")
    x_rolled = jnp.roll(x_cep, winding_tuple, (0, 1))
    x_unphased = lax.slice(x_rolled, (0, 0), support_shape)
    x_unphased = x_unphased / jnp.sign(x_unphased[*winding_tuple])
    return x_unphased


def fast_schwarz_transform(y, winding_tuple, support_shape):
    cepstrum = jnp.fft.ifft2(jnp.log(y))
    cep_shape = cepstrum.shape
    index_grid = jnp.indices(cep_shape)
    i_indices = index_grid[0, :, :]
    j_indices = index_grid[1, :, :]
    mask_winding = jnp.logical_and(
        jnp.less(i_indices, 2 * winding_tuple[0] + 1),
        jnp.less(j_indices, 2 * winding_tuple[1] + 1),
    )
    mask_trim = jnp.logical_and(
        jnp.less(i_indices, cep_shape[0] // 2), jnp.less(j_indices, cep_shape[1] // 2)
    )
    cepstrum = cepstrum.at[0, 0].multiply(0.5)
    cepstrum = jnp.roll(cepstrum, (2 * winding_tuple[0], 2 * winding_tuple[1]), (0, 1))
    cepstrum = jnp.where(mask_winding, cepstrum * 0.5, cepstrum)
    cepstrum = jnp.roll(cepstrum, (-winding_tuple[0], -winding_tuple[1]), (0, 1))
    cepstrum = jnp.where(mask_trim, cepstrum, 0.0)
    cepstrum = jnp.roll(cepstrum, (-winding_tuple[0], -winding_tuple[1]), (0, 1))
    x_cep = jnp.fft.ifft2(jnp.exp(jnp.fft.fft2(cepstrum)), norm="ortho")
    x_rolled = jnp.roll(x_cep, winding_tuple, (0, 1))
    x_unphased = x_rolled[0 : support_shape[0], 0 : support_shape[1]]
    x_unphased = x_unphased / jnp.sign(x_unphased[*winding_tuple])
    return x_unphased


@jax.jit
def refine(
    x_init: jax.typing.ArrayLike,  # Initial Guess for x
    far_field_intensities: jax.typing.ArrayLike,  # Far-Field Intensities
    support_mask: jax.typing.ArrayLike,  # Mask to indicate where x can be non-zero
    phase_reference_point: Tuple[int, int] = (
        0,
        0,
    ),  # Point to use as zero-phase reference
    scale_gradient: bool = False,  # Whether to scale the loss funciton such that the initial infinity norm of the gradient is no more than 1
    max_iters: int = 100,  # Maximum iterations for trust region minimization
    grad_tolerance: float = 1e-5,  # Gradient tolerance w/respect to infinity norm of gradient
    tv_reg: float = 0.0,
    far_field_mask: Optional[jax.typing.ArrayLike] = None,
    loss_type=L2_MAG_LOSS,
    cost_match_reg: float = 0.0,
    cost_val_2: float = 0.0,
    phase_reg: float = 1.0,
    descent_method: int = LBFGS,
) -> Tuple[jax.Array, float]:
    mask = support_mask
    support_shape = jnp.shape(mask)
    x_slice = lax.slice(x_init, (0, 0), support_shape)
    if far_field_mask is None:
        ff_mask = jnp.ones_like(far_field_intensities)
    else:
        ff_mask = far_field_mask

    x0 = view_as_flat_real(mask * x_slice)
    far_field_mags = jnp.sqrt(far_field_intensities)

    def L2_Loss(x):
        return masked_L2_mag_loss(
            x,
            far_field_mags,
            mask,
            x_slice.shape,
            phase_reference_point,
            tv_reg,
            ff_mask,
            phase_reg=phase_reg,
        )

    def Poisson_Loss(x):
        return masked_poisson_loss(
            x,
            far_field_mags,
            mask,
            x_slice.shape,
            phase_reference_point,
            tv_reg,
            ff_mask,
        )

    def Mag_Loss(x):
        return masked_mag_loss(
            x,
            far_field_mags,
            mask,
            x_slice.shape,
            phase_reference_point,
            tv_reg,
            ff_mask,
        )

    def L1_Loss(x):
        return masked_L1_mag_loss(
            x,
            far_field_mags,
            mask,
            x_slice.shape,
            phase_reference_point,
            tv_reg,
            ff_mask,
        )

    def loss_func(x):
        return lax.switch(loss_type, [L2_Loss, Poisson_Loss, Mag_Loss, L1_Loss], x)

    def true_fun():
        return 1 / jnp.fmax(
            jnp.linalg.vector_norm(jax.grad(loss_func)(x0), ord=jnp.inf), 1.0
        )

    def false_fun():
        return 1.0

    loss_scaling = 1.0

    def scaled_loss(x, dummy=None):
        return loss_scaling * loss_func(x)

    def min_trust_region(dummy=None):
        result = minimize_trust_region(scaled_loss, x0, max_iters, gtol=grad_tolerance)
        return (view_as_complex(result.x_k, support_shape), result.f_k)

    def minimize_lbfgs(dummy=None):
        result = lbfgs_minimize(
            scaled_loss, x0, max_iters=max_iters, rtol=grad_tolerance
        )
        x_c = view_as_complex(result.opt_params, support_shape)
        return (x_c, scaled_loss(result.opt_params))

    return lax.switch(descent_method, [min_trust_region, minimize_lbfgs], None)


def L2_mag_loss(x, y, shape, phase_ref_point, phase_reg=1):
    x_c = view_as_complex(x, shape)
    return (
        jnp.square(
            jnp.linalg.vector_norm(
                jnp.square(jnp.abs(jnp.fft.fft2(x_c, s=y.shape, norm="ortho"))) / y - y,
                ord=1,
            )
        )
        / 8
        + phase_reg * jnp.square(jnp.imag(x_c[*phase_ref_point])) / 2
    )


def masked_L2_mag_loss(
    x, y, mask, shape, phase_ref_point, tv_reg, ff_mask, phase_reg=1.0
):
    x_c = mask * view_as_complex(x, shape)
    y = jnp.fmax(y, 1e-14)
    gradient_1 = jnp.stack(jnp.gradient(x_c), axis=2)
    gradient_2 = jnp.linalg.vector_norm(jnp.abs(gradient_1) + 1e-12, axis=2)
    tv = jnp.linalg.vector_norm(gradient_2, ord=1)
    return (
        jnp.square(
            jnp.linalg.vector_norm(
                jnp.square(jnp.abs(jnp.fft.fft2(x_c, s=y.shape, norm="ortho"))) / y - y
            )
        )
        / 8
        + phase_reg * jnp.square(jnp.imag(x_c[*phase_ref_point])) / 2
        + tv_reg * tv
    )


def masked_poisson_loss(
    x, y, mask, shape, phase_ref_point, tv_reg, ff_mask, phase_reg=1
):
    x_c = mask * view_as_complex(x, shape)
    intensities = jnp.fmax(
        jnp.square(jnp.abs(jnp.fft.fft2(x_c, s=y.shape, norm="ortho"))), 1e-14
    )
    gradient_1 = jnp.stack(jnp.gradient(x_c), axis=2)
    gradient_2 = jnp.linalg.vector_norm(jnp.abs(gradient_1) + 1e-12, axis=2)
    tv = jnp.linalg.vector_norm(gradient_2, ord=1)
    return (
        jnp.sum((intensities - jnp.square(y) * jnp.log(intensities)))
        + phase_reg * jnp.square(jnp.imag(x_c[*phase_ref_point])) / 2
        + tv * tv_reg
    )


def masked_mag_loss(x, y, mask, shape, phase_ref_point, tv_reg, ff_mask, phase_reg=1):
    x_c = mask * view_as_complex(x, shape)
    y = jnp.fmax(y, 1e-14)
    gradient_1 = jnp.stack(jnp.gradient(x_c), axis=2)
    gradient_2 = jnp.linalg.vector_norm(jnp.abs(gradient_1) + 1e-12, axis=2)
    tv = jnp.linalg.vector_norm(gradient_2, ord=1)
    return (
        jnp.square(
            jnp.linalg.vector_norm(
                jnp.square(jnp.abs(jnp.fft.fft2(x_c, s=y.shape, norm="ortho")))
                - jnp.square(y)
            )
        )
        / 8
        + phase_reg * jnp.square(jnp.imag(x_c[*phase_ref_point])) / 2
        + tv_reg * tv
    )


def masked_L1_mag_loss(
    x, y, mask, shape, phase_ref_point, tv_reg, ff_mask, phase_reg=1e-10
):
    x_c = mask * view_as_complex(x, shape)
    y = jnp.fmax(y, 1e-14)
    gradient_1 = jnp.stack(jnp.gradient(x_c), axis=2)
    gradient_2 = jnp.linalg.vector_norm(jnp.abs(gradient_1) + 1e-12, axis=2)
    tv = jnp.linalg.vector_norm(gradient_2, ord=1)
    return (
        jnp.linalg.vector_norm(
            (jnp.square(jnp.abs(jnp.fft.fft2(x_c, s=y.shape, norm="ortho"))) / y - y),
            ord=1,
        )
        / 8
        + phase_reg * jnp.square(jnp.imag(x_c[*phase_ref_point])) / 2
        + tv_reg * tv
    )


def dummy_func(scaled_loss, x0, max_iters, gtol=1e-5):
    result = minimize_trust_region(scaled_loss, x0, max_iters, gtol=gtol)
    return None
