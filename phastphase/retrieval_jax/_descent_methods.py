from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import lax


class _LBFGSState(NamedTuple):
    opt_state: optax.OptState
    opt_params: jnp.ndarray
    iteration: int
    grad_norm: float


def _lbfgs_cond_func(
    state: _LBFGSState, abs_tol: float, rel_tol: float, max_iters: int, g0: float
) -> jnp.ndarray:
    """Condition function for the LBFGS loop."""
    return jnp.logical_and(
        jnp.less(state.iteration, max_iters),
        jnp.logical_and(
            jnp.greater(state.grad_norm, abs_tol),
            jnp.greater(state.grad_norm, rel_tol * g0),
        ),
    )


def _lbfgs_step(
    state: _LBFGSState,
    cost_function: Callable,
    value_and_grad: Callable,
    lbfgs_solver: optax.GradientTransformationExtraArgs,
) -> _LBFGSState:
    """Perform a single step of the LBFGS optimization algorithm."""
    value, grad = value_and_grad(state.opt_params, state=state.opt_state)
    updates, new_opt_state = lbfgs_solver.update(
        grad,
        state.opt_state,
        state.opt_params,
        value=value,
        grad=grad,
        value_fn=cost_function,
    )
    new_params = optax.apply_updates(state.opt_params, updates)
    return _LBFGSState(
        opt_state=new_opt_state,
        opt_params=new_params,
        iteration=state.iteration + 1,
        grad_norm=jnp.linalg.norm(grad),
    )


def lbfgs_minimize(
    cost_function: Callable,
    x_0: jnp.ndarray,
    max_itertaions: int,
    rtol: float = 1e-5,
    atol: float = 1e-15,
) -> _LBFGSState:
    """Optimize a function using the LBFGS algorithm."""
    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(cost_function)
    g0 = jnp.linalg.norm(value_and_grad(x_0, state=solver.init(x_0))[1])
    state = _LBFGSState(
        opt_state=solver.init(x_0),
        opt_params=x_0,
        iteration=0,
        grad_norm=g0,
    )
    cond_func = jax.tree_util.Partial(
        _lbfgs_cond_func, abs_tol=atol, rel_tol=rtol, max_iters=max_itertaions, g0=g0
    )
    step_func = jax.tree_util.Partial(
        _lbfgs_step,
        cost_function=cost_function,
        value_and_grad=value_and_grad,
        lbfgs_solver=solver,
    )
    final_state = lax.while_loop(cond_func, step_func, state)
    return final_state


def lbfgs_saddlepoint_escape(
    cost_function: Callable,
    x_0: jnp.ndarray,
    max_itertaions: int,
    rtol: float = 1e-5,
    atol: float = 1e-15,
):
    """Optimize a function using a compound LBFGS/ Newton method."""
