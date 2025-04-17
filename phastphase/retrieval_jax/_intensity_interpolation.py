import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize


# @partial(jnp.vectorize, excluded={0}, signature='()->(n)')
def ifft_vec(support, index):
    unit_vector = jnp.zeros_like(support)
    unit_vector = unit_vector.at[index].set(1)
    return jnp.ravel(jnp.fft.ifft2(unit_vector))


def interpolation_operator(support_mask, missing_intensity_indices, autocorr_indices):
    P = jnp.zeros(
        (jnp.size(autocorr_indices), len(missing_intensity_indices[0])),
        dtype=jnp.complex128,
    )
    missing_index_list = [
        (missing_intensity_indices[0][i], missing_intensity_indices[1][i])
        for i in range(len(missing_intensity_indices[0]))
    ]
    for col_index, intensity_index in enumerate(missing_index_list):
        P = P.at[:, col_index].set(
            ifft_vec(support_mask, intensity_index)[autocorr_indices]
        )
    return P


def calculate_residual(intensities, autocorr_indices):
    autocorr = jnp.fft.ifft2(intensities)
    return autocorr[autocorr_indices]


def calc_flat_autocorr_indices(support_mask):
    intensity = jnp.square(jnp.abs(jnp.fft.fft2(support_mask)))
    autocorr = jnp.fft.ifft2(intensity)
    mask = jnp.ones_like(autocorr)
    mask = mask.at[autocorr > 0].set(0)
    return jnp.flatnonzero(mask)


def calc_autocorr_indices_choice(support_mask, num_points):
    intensity = jnp.square(jnp.abs(jnp.fft.fft2(support_mask)))
    autocorr = jnp.fft.ifft2(intensity)
    mask = jnp.ones_like(autocorr)
    mask = mask.at[autocorr > 0.5].set(0)
    flat_indices = jnp.flatnonzero(mask)
    key = jax.random.key(42)
    chosen_flat_indices = jax.random.choice(key, flat_indices, (num_points,), False)
    chosen_non_flat_indices = jnp.unravel_index(chosen_flat_indices, mask.shape)
    return (chosen_non_flat_indices, chosen_flat_indices)


def calc_autocorr_indices(support_mask):
    intensity = jnp.square(jnp.abs(jnp.fft.fft2(support_mask)))
    autocorr = jnp.fft.ifft2(intensity)
    mask = jnp.ones_like(autocorr)
    mask = mask.at[autocorr > 0.5].set(0)
    flat_indices = jnp.flatnonzero(mask)
    non_flat_indices = jnp.unravel_index(flat_indices, mask.shape)
    return (non_flat_indices, flat_indices)


def interpolate_intensities(
    far_field_intensities,  # Total far-field pattern, with missing elements set to 0
    near_field_support_mask,  # Support mask, the total size should be the size of the far-field
    missing_intensity_indices,  # indices of the missing intensity measurements. Should of the form you get from jax.numpy.nonzero
    num_matrix_rows,
    non_neg_leat_squares: bool = True,
):
    autocorr_indices, flat_autocorr_indices = calc_autocorr_indices_choice(
        near_field_support_mask, num_matrix_rows
    )
    b_complex = -calculate_residual(far_field_intensities, autocorr_indices)
    interpolation_matrix_complex = interpolation_operator(
        near_field_support_mask, missing_intensity_indices, flat_autocorr_indices
    )
    b_real_imag = jnp.concat([jnp.real(b_complex), jnp.imag(b_complex)])
    A_matrix = jnp.concat(
        [jnp.real(interpolation_matrix_complex), jnp.imag(interpolation_matrix_complex)]
    )
    b_numpy = np.asarray(b_real_imag)
    A_numpy = np.asarray(A_matrix)
    if non_neg_leat_squares:
        x = non_negative_least_squares(A_numpy, b_numpy)
    else:
        x = standard_least_squares(A_numpy, b_numpy)
    return jnp.asarray(x)


def replace_negative_intensities(
    far_field_intensities,
    near_field_support_mask,
    num_matrix_rows,
    non_neg_least_squares: bool = True,
):
    index_mask = jnp.zeros_like(far_field_intensities)
    index_mask = index_mask.at[jnp.less(far_field_intensities, 0)].set(1)
    missing_indices = jnp.nonzero(index_mask)
    far_field_intensities = far_field_intensities.at[missing_indices].set(0)
    missing_intensities = interpolate_intensities(
        far_field_intensities,
        near_field_support_mask,
        missing_indices,
        num_matrix_rows,
        non_neg_least_squares,
    )

    far_field_intensities = far_field_intensities.at[missing_indices].set(
        missing_intensities
    )
    return far_field_intensities


def interpolation_linear_operator(
    missing_intensities, full_intensities, missing_intensity_indices, autocorr_indices
):
    full_intensities = full_intensities.at[missing_intensity_indices].set(
        missing_intensities, unique_indices=True
    )
    autocorr = jnp.fft.ifft2(full_intensities)
    autocorr = autocorr[autocorr_indices]
    real_autocorr = jnp.real(autocorr)
    imag_autocorr = jnp.imag(autocorr)

    return jnp.concat([real_autocorr, imag_autocorr])


def matrix_free_intensity_interpolation(
    far_field_intensities,  # Total far-field pattern, with missing elements set to 0
    near_field_support_mask,  # Support mask, the total size should be the size of the far-field
    missing_intensity_indices,  # indices of the missing intensity measurements. Should of the form you get from jax.numpy.nonzero)
    rtol: float = 1e-9,
    atol: float = 1e-8,
):
    autocorr_indices, flat_autocorr_indices = calc_autocorr_indices(
        near_field_support_mask
    )
    b_complex = -calculate_residual(far_field_intensities, autocorr_indices)
    b_real_imag = jnp.concat([jnp.real(b_complex), jnp.imag(b_complex)])
    full_intensities = jnp.zeros_like(far_field_intensities)
    non_flat_indices = jnp.unravel_index(
        missing_intensity_indices, jnp.shape(full_intensities)
    )

    def interp_operator(x):
        return interpolation_linear_operator(
            x, full_intensities, non_flat_indices, autocorr_indices
        )

    dummy = jnp.zeros(jnp.size(missing_intensity_indices))
    linear_operator = lx.FunctionLinearOperator(
        interp_operator, jax.eval_shape(lambda: dummy)
    )
    # solver = LSMR(rtol, atol)
    solver = lx.NormalCG(rtol, atol)
    out = lx.linear_solve(linear_operator, b_real_imag, solver)
    return out.value


def mat_free_replace_negative_intensities(
    far_field_intensities,
    near_field_support_mask,
    rtol: float = 1e-12,
    atol: float = 1e-12,
):
    index_mask = jnp.zeros_like(far_field_intensities)
    index_mask = index_mask.at[jnp.less(far_field_intensities, 0)].set(1)
    missing_indices = jnp.flatnonzero(index_mask)
    non_flat_indices = jnp.nonzero(index_mask)
    far_field_intensities = far_field_intensities.at[non_flat_indices].set(0)
    missing_intensities = matrix_free_intensity_interpolation(
        far_field_intensities, near_field_support_mask, missing_indices, rtol, atol
    )
    far_field_intensities = far_field_intensities.at[non_flat_indices].set(
        missing_intensities
    )
    return far_field_intensities


def standard_least_squares(A, b):
    x, res, rank, svds = scipy.linalg.lstsq(A, b)
    return x


def non_negative_least_squares(A, b):
    x, rnorm = scipy.optimize.nnls(A, b)
    return x
