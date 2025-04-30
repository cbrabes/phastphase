"""
Implementation of Algorithm 1 in:
"A Newton-CG Algorithm with Complexity Guarantees
for Smooth Unconstrained Optimization"


"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.experimental
import jax.experimental.sparse
import jax.numpy as jnp
from jax import lax
from jax.tree_util import Partial
from jax.typing import ArrayLike

_dot = Partial(jnp.dot, precision=lax.Precision.HIGHEST)


def _get_boundaries_intersection(
    z: jnp.ndarray, d: jnp.ndarray, trust_radius: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """
    ported from scipy

    Solve the scalar quadratic equation ||z + t d|| == trust_radius.
    This is like a line-sphere intersection.
    Return the unique positive value of t.
    The values of z,d, and trust_radius when used in our trust region method implies the equation possesses two real roots of opposite sign.
    The positive root is the one that is relevant for the trust region subproblem.
    """
    a = _dot(d, d)
    b = 2 * _dot(z, d)
    c = _dot(z, z) - trust_radius**2
    sqrt_discriminant = jnp.sqrt(b * b - 4 * a * c)

    # The following calculation is mathematically
    # equivalent to:
    # ta = (-b - sqrt_discriminant) / (2*a)
    # tb = (-b + sqrt_discriminant) / (2*a)
    # but produce smaller round off errors.
    # Look at Matrix Computation p.97
    # for a better justification.
    aux = b + jnp.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux

    # ta and tb are real and opposite sign, therefore the greater value is the positive root
    ra = jnp.where(ta < tb, ta, tb)
    rb = jnp.where(ta < tb, tb, ta)
    return (ra, rb)


# Convergence Status
# 0: negative curvature
# 1: hit boundary
# 2: interior solution
# 3: iteration limit reached
NOT_CONVERGED = -1
NEGATIVE_CURVATURE = 0
HIT_BOUNDARY = 1
INTERIOR_SOLUTION = 2
ITERATION_LIMIT_REACHED = 3


class _CappedCGState(NamedTuple):
    residual: jnp.ndarray
    residual_norm_squared: float
    p: jnp.ndarray
    y: jnp.ndarray
    step_number: int
    convergence_status: ArrayLike
    converged: ArrayLike
    search_direction: jnp.ndarray


def _capped_cg_step(
    state: _CappedCGState,
    hvp_func: Callable,
    relative_accuracy: float,
    g_norm: ArrayLike,
    iter_limit: int,
    trust_radius: float,
    second_order_change_f: Callable,
) -> _CappedCGState:
    # Perform Standard CG Step
    hess_p = hvp_func(state.p)
    alpha = state.residual_norm_squared / _dot(state.p, hess_p)
    y_new = state.y + alpha * state.p
    r_new = state.residual + alpha * hess_p
    r_new_norm_squared = _dot(r_new, r_new)
    beta = r_new_norm_squared / state.residual_norm_squared
    p_new = -r_new + beta * state.p

    # End Standard CG Steo
    r_a, r_b = _get_boundaries_intersection(state.y, state.p, trust_radius)

    search_a = state.y + r_a * state.p

    search_b = state.y + r_b * state.p

    # List of Possible Search Direction
    search_direction_neg_curve = state.y + r_b * state.p
    search_direction_hit_boundary = jnp.where(
        second_order_change_f(search_a) < second_order_change_f(search_b),
        search_a,
        search_b,
    )
    search_direction_interior = y_new
    search_direction_iter_limit = y_new

    # The convergence criteria:

    # Negative Curvature Detected
    negative_curvature = jnp.less(_dot(state.p, hess_p), 0)

    # Search Direction Hits Boundary
    hit_boundary = jnp.greater_equal(jnp.linalg.vector_norm(y_new), trust_radius)

    # CG has converged to the interior of the trust region
    interior_solution = jnp.less_equal(
        jnp.linalg.vector_norm(r_new),
        (relative_accuracy / 2) * g_norm,
    )

    # Limit of cg iterations reached
    iter_limit_reached = jnp.greater_equal(state.step_number + 1, iter_limit)

    # If any are true, return a search direction
    converged = jnp.logical_or(
        interior_solution,
        jnp.logical_or(
            negative_curvature, jnp.logical_or(hit_boundary, iter_limit_reached)
        ),
    )

    # Convergence Status
    # 0: negative curvature
    # 1: hit boundary
    # 2: interior solution
    # 3: iteration limit reached
    convergence_status = jnp.where(
        negative_curvature,
        0,
        jnp.where(
            hit_boundary,
            1,
            jnp.where(interior_solution, 2, jnp.where(iter_limit_reached, 3, -1)),
        ),
    )

    # Select the Appropriate Search Direction
    search_direction = lax.select_n(
        convergence_status,
        search_direction_neg_curve,
        search_direction_hit_boundary,
        search_direction_interior,
        search_direction_iter_limit,
    )
    return _CappedCGState(
        residual=r_new,
        residual_norm_squared=r_new_norm_squared,
        p=p_new,
        y=y_new,
        step_number=state.step_number + 1,
        convergence_status=convergence_status,
        converged=converged,
        search_direction=search_direction,
    )


def _capped_cg_cond(state: _CappedCGState) -> ArrayLike:
    return jnp.logical_not(state.converged)


def capped_cg_method(
    hessvp: Callable,
    gradient: jnp.ndarray,
    trust_radius: ArrayLike,
    relative_accuracy: float,
    iter_limit: ArrayLike,
) -> _CappedCGState:
    """
    Conjugate gradient method for solving the trust region subproblem.
    """
    r0 = gradient
    r0_norm_squared = _dot(r0, r0)
    p0 = -r0
    y0 = jnp.zeros_like(gradient)
    init_state = _CappedCGState(
        residual=r0,
        residual_norm_squared=r0_norm_squared,
        p=p0,
        y=y0,
        step_number=0,
        convergence_status=jnp.array(-1),
        converged=jnp.array(False),
        search_direction=jnp.zeros_like(gradient),
    )
    _capped_body_fun = Partial(
        _capped_cg_step,
        hvp_func=hessvp,
        relative_accuracy=relative_accuracy,
        g_norm=jnp.linalg.vector_norm(gradient),
        iter_limit=iter_limit,
        trust_radius=trust_radius,
    )
    final_state: _CappedCGState = lax.while_loop(
        _capped_cg_cond,
        _capped_body_fun,
        init_state,
    )
    return final_state


class _LanczosState(NamedTuple):
    w: jnp.ndarray
    v: jnp.ndarray
    beta_vector: jnp.ndarray
    alpha_vector: jnp.ndarray


def _lanczos_body_fun(
    iter: int, state: _LanczosState, hessvp: Callable
) -> _LanczosState:
    """
    The body function for the Lanczos method.
    """
    # Compute the next Lanczos vector
    beta = jnp.linalg.vector_norm(state.w)
    v = state.w / beta
    wp = hessvp(v)
    alpha = _dot(v, wp)
    w = wp - alpha * v - beta * state.v

    new_betas_vector = state.beta_vector.at[iter].set(beta)
    new_alphas_vector = state.alpha_vector.at[iter].set(alpha)
    return _LanczosState(
        w=w,
        v=v,
        beta_vector=new_betas_vector,
        alpha_vector=new_alphas_vector,
    )


def lanczos_iteration(
    hessvp: Callable,
    v0: jnp.ndarray,
    maxiter: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    The Lanczos method for tridiagonalization of a hessian.
    """

    wp1 = hessvp(v0)
    alpha0 = _dot(v0, wp1)
    w = wp1 - alpha0 * v0
    init_state = _LanczosState(
        w=w,
        v=v0,
        beta_vector=jnp.zeros(maxiter),
        alpha_vector=jnp.zeros(maxiter),
    )
    final_state: _LanczosState = lax.fori_loop(
        0, maxiter, _lanczos_body_fun, init_state
    )
    output_alphas = jnp.insert(
        final_state.alpha_vector,
        0,
        alpha0,
    )

    return (output_alphas, final_state.beta_vector)


def min_eig_lanczos(
    hessvp,
    v0: jnp.ndarray,
    maxiter: int,
) -> ArrayLike:
    """
    The Lanczos method for tridiagonalization of a hessian.
    """
    alphas, betas = lanczos_iteration(hessvp, v0, maxiter)
    # Compute the eigenvalues of the tridiagonal matrix
    eigvals = jax.scipy.linalg.eigh_tridiagonal(alphas, betas, eigvals_only=True)
    min_eigval = jnp.min(eigvals)
    return min_eigval


def randomized_min_lanczos(
    hessvp,
    example_x: jnp.ndarray,
    maxiter: int,
    random_key: int = 42,
) -> ArrayLike:
    key = jax.random.key(random_key)
    v0 = jax.random.normal(key, example_x.shape)
    alphas, betas = lanczos_iteration(hessvp, v0, maxiter)
    # Compute the eigenvalues of the tridiagonal matrix
    eigvals = jax.scipy.linalg.eigh_tridiagonal(alphas, betas, eigvals_only=True)
    min_eigval = jnp.min(eigvals)
    return min_eigval


def lobpcg_min_eigpair(
    hvp: Callable,
    example_x: jnp.ndarray,
    iter_limit: int,
    tol: float = 1e-10,
    random_key: int = 42,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Returns the minimum eigenvalue and eigenvector of the Hessian
    using the LOBPCG method.

    """
    key = jax.random.key(random_key)

    X0 = jax.random.normal(key, example_x.shape)
    X0 = X0 / jnp.linalg.vector_norm(X0)

    def negative_hvp(p: jnp.ndarray) -> jnp.ndarray:
        return -hvp(p)

    (eigval, eigvecs, iter_reached) = jax.experimental.sparse.linalg.lobpcg_standard(
        negative_hvp, X0, tol=tol, m=iter_limit
    )
    return (eigvecs, -eigval)
