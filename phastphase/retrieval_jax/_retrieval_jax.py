from functools import partial
from typing import Optional, Tuple

from tqdm import tqdm

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

WIND_METHOD_ITERATIVE: int = 0

WIND_METHOD_CONVOLUTION: int = 1


@jax.jit
def retrieve(
    far_field_intensities: jax.typing.ArrayLike,  # Magnitude squared of fourier transform
    support_mask: jax.typing.ArrayLike,  # Support mask
    winding_guess=None,  # Guess for the winding number
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
    wind_method: int = WIND_METHOD_ITERATIVE
) -> Tuple[jax.Array, float]:  # Loss Type
    mask = support_mask
    support_shape = mask.shape

    if far_field_mask is None:
        ff_mask = jnp.ones_like(far_field_intensities)
    else:
        ff_mask = far_field_mask

    offset_intensities = far_field_intensities + offset
    cepstrum = jnp.fft.ifft2(jnp.log(offset_intensities))

    # Required to ensure both lambda functions have the same return type,
    # it is not smart enough to understand that if the false lambda is executed, winding_guess will never be None.
    winding_guess_jax = (0, 0) if winding_guess is None else winding_guess

    winding_number = lax.cond(
        winding_guess is None,
        lambda _: winding_calc(support_mask, offset_intensities, cepstrum, wind_method=wind_method),
        lambda _: winding_guess_jax,
        operand=None
    )

    jax.debug.print("Using winding numbers: ({wind_x}, {wind_y})", wind_x = winding_number[0], wind_y = winding_number[1])

    x_schwarz = fast_schwarz_transform(
        cepstrum, winding_number, support_shape
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
    support_mask: jnp.ndarray, 
    far_field_intensities: jnp.ndarray,
    cepstrum: jnp.ndarray,
    wind_method: int = 0
) -> Tuple[jax.typing.ArrayLike, ...]:
    
    branches = (
        lambda _: iterative_winding_calc(support_shape=support_mask.shape, cepstrum=cepstrum),
        lambda _: convolution_winding_calc(support_mask=support_mask, far_field_intensities=far_field_intensities),
    )

    # Dispatch by integer method ID
    return lax.switch(wind_method, branches, operand=None)


def convolution_winding_calc(
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


def last_nonzero_index(data: jnp.ndarray, axis: int) -> int:
    return jnp.where(jnp.heaviside(data,0), size=data.size)[axis].max(initial=0)


def iterative_winding_calc_single_axis(
    support_shape: tuple[int, ...],
    wrap_section_shape: tuple[int, ...], 
    cepstrum: jnp.ndarray, 
    main_quadrant: jnp.ndarray,
    axis: int,
    num_loops: int = 100
) -> int:
    support_len = support_shape[axis]
    init_cepstrum_high = jnp.max(jnp.abs(cepstrum))
    init_cepstrum_low = jnp.min(jnp.abs(cepstrum))
    wrap_section = jnp.abs(cepstrum[wrap_section_shape])

    def loop_body(i, state):
        curr_cepstrum_high, curr_cepstrum_low, curr_winding_num, curr_done = state
        
        def loop_step(_):
            threshhold = (curr_cepstrum_high+curr_cepstrum_low)/2.

            wraparound_last_nonzero = last_nonzero_index(wrap_section - threshhold, axis=axis)
            main_last_nonzero = last_nonzero_index(main_quadrant - threshhold, axis=axis)
            length = (wraparound_last_nonzero + main_last_nonzero + 1)
        
            next_cepstrum_high = lax.cond(length < support_len, lambda _: threshhold, lambda _: curr_cepstrum_high, operand=None)
            next_cepstrum_low = lax.cond(length > support_len, lambda _: threshhold, lambda _: curr_cepstrum_low, operand=None)
            next_winding_num = wraparound_last_nonzero
            next_done = (length == support_len)

            return (next_cepstrum_high, next_cepstrum_low, next_winding_num, next_done)

        return lax.cond(curr_done, lambda _: state, loop_step, operand=None)
    
    # Initialize state variables
    init_state = (init_cepstrum_high, init_cepstrum_low, 0, False)

    # Execute for loop
    final_high, final_low, winding_num, done = lax.fori_loop(0, num_loops, loop_body, init_state)

    return winding_num


def iterative_winding_calc(
    support_shape: tuple[int, ...],
    cepstrum: jnp.ndarray,
    num_loops: int = 100
) -> tuple[int]:
    n = support_shape[0]
    m = support_shape[1]

    upper_right_corner_slice = (slice(0, n//2), slice(-m,-m//2+1))
    bottom_left_corner_slice = (slice(-n, -n//2+1), slice(0, m//2+1))

    main_quadrant = jnp.abs(cepstrum[0:n,0:m])

    x_winding_number = iterative_winding_calc_single_axis(
        support_shape=support_shape,
        wrap_section_shape=upper_right_corner_slice,
        cepstrum=cepstrum,
        main_quadrant=main_quadrant,
        axis=0,
        num_loops=num_loops
    )

    y_winding_number = iterative_winding_calc_single_axis(
        support_shape=support_shape,
        wrap_section_shape=bottom_left_corner_slice,
        cepstrum=cepstrum,
        main_quadrant=main_quadrant,
        axis=1,
        num_loops=num_loops
    )

    return (x_winding_number, y_winding_number)


def schwarz_transform(cepstrum, winding_tuple, support_shape):
    cep_mask = jnp.zeros_like(cepstrum)
    fft_freqs_axis0 = jnp.fft.fftfreq(cepstrum.shape[0], d=1.0 / cepstrum.shape[0])
    fft_freqs_axis1 = jnp.fft.fftfreq(cepstrum.shape[1], d=1.0 / cepstrum.shape[1])
    x_grid, y_grid = jnp.meshgrid(fft_freqs_axis0, fft_freqs_axis1, indexing="ij")

    cep_mask = cep_mask.at[0 : cepstrum.shape[0] // 2, 0 : cepstrum.shape[1] // 2].set(1)
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


def fast_schwarz_transform(cepstrum, winding_tuple, support_shape):
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
