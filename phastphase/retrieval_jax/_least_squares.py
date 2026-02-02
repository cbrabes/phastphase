from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import lineax as lx


class _LevenbergMarquardtState(NamedTuple):
    x: jnp.ndarray
    rho: float
    rtol: float
    atol: float
    stabilize_r_iters: int
    max_cg_iters: int
    step_number: int
    residuals: jnp.ndarray


def least_squares_linear_operator(x, jvp_function: Callable, damping):
    jvp = jvp_function(x)
    damped_term = damping * x
    return jnp.concatenate((jvp, damped_term))


def _levenber_marquadt_cond(
    state: _LevenbergMarquardtState, tolerance: float, max_iters: int
) -> jnp.ndarray:
    r_norm = jnp.linalg.vector_norm(state.residuals)
    return jnp.logical_and(
        jnp.greater(r_norm, tolerance), jnp.less(state.step_number, max_iters)
    )


def _levenberg_marquadt_step(
    state: _LevenbergMarquardtState, residual_function: Callable
) -> _LevenbergMarquardtState:
    residuals, jvp_func = jax.linearize(residual_function, state.x)

    least_squares_operator = jax.tree_util.Partial(
        least_squares_linear_operator,
        jvp_function=jvp_func,
        damping=state.rho * jnp.linalg.vector_norm(residuals),
    )
    lx_operator = lx.FunctionLinearOperator(
        least_squares_operator, jax.eval_shape(lambda: state.x)
    )
    solver = lx.NormalCG(
        atol=state.atol,
        rtol=state.rtol,
        stabilise_every=state.stabilize_r_iters,
        max_steps=state.max_cg_iters,
    )
    rhs = jnp.concatenate((-residuals, jnp.zeros_like(state.x)))

    result = lx.linear_solve(lx_operator, rhs, solver=solver)
    new_x = state.x + result.value  # Update the solution
    return _LevenbergMarquardtState(
        x=new_x,
        rho=state.rho,
        residuals=residual_function(new_x),  # Recompute residuals for the new x
        rtol=state.rtol,
        atol=state.atol,
        stabilize_r_iters=state.stabilize_r_iters,
        max_cg_iters=state.max_cg_iters,
        step_number=state.step_number + 1,
    )


def levenberg_marquadt(
    x0: jnp.ndarray,
    residual_func: Callable,
    max_iters: int = 100,
    init_rho: float = 1e-4,
    tol: float = 1e-6,
    cg_rtol: float = 1e-6,
    cg_atol: float = 1e-6,
    stabilize_r_iters: int = 10,
    max_cg_iters: int = 10000,
) -> _LevenbergMarquardtState:
    """Levenberg-Marquardt algorithm for nonlinear least squares.

    Args:
        x0 (jax.Array): Initial guess.
        cost_func (callable): Cost function that returns residuals.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        lambda_init (float): Initial damping factor.

    Returns:
        jax.Array: Solution vector.
    """
    step_func = jax.tree_util.Partial(
        _levenberg_marquadt_step,
        residual_function=residual_func,
    )
    init_state = _LevenbergMarquardtState(
        x=x0,
        rho=init_rho,
        residuals=residual_func(x0),  # Compute initial residuals
        atol=cg_atol,
        rtol=cg_rtol,
        stabilize_r_iters=stabilize_r_iters,
        step_number=0,
        max_cg_iters=max_cg_iters,
    )

    cond_function = jax.tree_util.Partial(
        _levenber_marquadt_cond, tolerance=tol, max_iters=max_iters
    )
    state = init_state
    while cond_function(state):
        state = step_func(state)
    return state


def phase_retrieval_levenberg_marquadt():
    return None
