from typing import NamedTuple

import jax.numpy as jnp
from jax import lax


class _ConditionState(NamedTuple):
    start_index: int
    best_start: int
    best_cond: float


def _condition_loop_func(
    state: _ConditionState, flat_y: jnp.ndarray, slice_size: int
) -> _ConditionState:
    slice = lax.dynamic_slice(flat_y, (state.start_index,), (slice_size,))
    cond = slice[-1] / slice[0] if slice[0] != 0 else jnp.inf  # Avoid division by zero
    new_best_cond, new_start = lax.cond(
        jnp.less(cond, state.best_cond),
        lambda: (cond, state.start_index),
        lambda: (state.best_cond, state.best_start),
    )
    new_state = _ConditionState(
        start_index=state.start_index + 1, best_start=new_start, best_cond=new_best_cond
    )
    return new_state


def best_conditioned_set(y: jnp.ndarray, slice_size: int) -> jnp.ndarray:
    sorted_indices = jnp.argsort(y, axis=None)
    y_vals = jnp.sort(y, axis=None)
    n = y_vals.size - slice_size + 1

    initial_state = _ConditionState(start_index=0, best_start=0, best_cond=jnp.inf)
    final_state = lax.fori_loop(0, n, _condition_loop_func, initial_state)
    return lax.dynamic_slice(sorted_indices, (final_state,), (slice_size,))
    # Return the best conditioned set of indices
