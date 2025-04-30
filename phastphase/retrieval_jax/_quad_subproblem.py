# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Callable, NamedTuple, Tuple, Union

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
from jax import lax

__all__ = ["CGSteihaugSubproblem"]


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
HessVP = Callable[[jnp.ndarray], jnp.ndarray]


class _QuadSubproblemResult(NamedTuple):
    step: jnp.ndarray
    hits_boundary: Union[bool, jnp.ndarray]
    pred_f: Union[float, jnp.ndarray]
    nfev: Union[int, jnp.ndarray]
    ngev: Union[int, jnp.ndarray]
    nhev: Union[int, jnp.ndarray]
    success: Union[bool, jnp.ndarray]


class _CGSteihaugState(NamedTuple):
    z: Union[float, jnp.ndarray]
    r: Union[float, jnp.ndarray]
    d: Union[float, jnp.ndarray]
    step: Union[float, jnp.ndarray]
    hits_boundary: Union[bool, jnp.ndarray]
    converged: Union[bool, jnp.ndarray]


def second_order_approx(
    p: jnp.ndarray,
    cur_val: Union[float, jnp.ndarray],
    g: jnp.ndarray,
    hessvp: HessVP,
) -> Union[float, jnp.ndarray]:
    return cur_val + _dot(g, p) + 0.5 * _dot(p, hessvp(p))


def get_boundaries_intersections(
    z: jnp.ndarray, d: jnp.ndarray, trust_radius: Union[float, jnp.ndarray]
):
    """
    ported from scipy

    Solve the scalar quadratic equation ||z + t d|| == trust_radius.
    This is like a line-sphere intersection.
    Return the two values of t, sorted from low to high.
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

    # (ta, tb) if ta < tb else (tb, ta)
    ra = jnp.where(ta < tb, ta, tb)
    rb = jnp.where(ta < tb, tb, ta)
    return (ra, rb)


def CGSteihaugSubproblem(
    cur_val: Union[float, jnp.ndarray],
    g: jnp.ndarray,
    g_mag: Union[float, jnp.ndarray],
    hessvp: HessVP,
    trust_radius: Union[float, jnp.ndarray],
    norm=jnp.inf,
) -> _QuadSubproblemResult:
    """
    Solve the subproblem using a conjugate gradient method.
    Parameters
    ----------
    cur_val : Union[float, jnp.ndarray]
      Objective value evaluated at the current state.
    g : jnp.ndarray
      Gradient value evaluated at the current state.
    g_mag : Union[float, jnp.ndarray]
      The magnitude of the gradient `g` using norm=`norm`.
    hessvp: Callable
      Function that accepts a proposal vector and computes the result of a
      Hessian-vector product.
    trust_radius : float
      Upper bound on how large a step proposal can be.
    norm : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
      Order of the norm. inf means jax.numpy’s inf object. The default is inf.
    Returns
    -------
    result : _QuadSubproblemResult
      Contains the step proposal, whether it is at radius boundary, and
      meta-data regarding function calls and successful convergence.

    Notes
    -----
    This is algorithm (7.2) of Nocedal and Wright 2nd edition.
    Only the function that computes the Hessian-vector product is required.
    The Hessian itself is not required, and the Hessian does
    not need to be positive semidefinite.
    """

    # second-order taylor series approximation at the current values, gradient, and hessian
    def soa(p):
        return second_order_approx(p, cur_val, g, hessvp)

    # these next few functons are helpers for internal switches in the main CGSteihaug logic
    def noop(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]],
    ) -> _CGSteihaugState:
        iterp, z_next = param
        return iterp

    def step1(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]],
    ) -> _CGSteihaugState:
        iterp, z_next = param
        ta, tb = get_boundaries_intersections(iterp.z, iterp.d, trust_radius)
        pa = iterp.z + ta * iterp.d
        pb = iterp.z + tb * iterp.d
        p_boundary = jnp.where(soa(pa) < soa(pb), pa, pb)
        return iterp._replace(step=p_boundary, hits_boundary=True, converged=True)

    def step2(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]],
    ) -> _CGSteihaugState:
        iterp, z_next = param
        ta, tb = get_boundaries_intersections(iterp.z, iterp.d, trust_radius)
        p_boundary = iterp.z + tb * iterp.d
        return iterp._replace(step=p_boundary, hits_boundary=True, converged=True)

    def step3(
        param: Tuple[_CGSteihaugState, Union[float, jnp.ndarray]],
    ) -> _CGSteihaugState:
        iterp, z_next = param
        return iterp._replace(step=z_next, hits_boundary=False, converged=True)

    # initialize the step
    p_origin = jnp.zeros_like(g)

    # define a default tolerance
    tolerance = jnp.min(jnp.array([0.5, jnp.sqrt(g_mag)])) * g_mag

    # init the state for the first iteration
    z = p_origin
    r = g
    d = -r
    init_param = _CGSteihaugState(
        z=z, r=r, d=d, step=p_origin, hits_boundary=False, converged=False
    )

    # Search for the min of the approximation of the objective function.
    def body_f(iterp: _CGSteihaugState) -> _CGSteihaugState:
        # do an iteration
        Bd = hessvp(iterp.d)
        dBd = _dot(iterp.d, Bd)

        # after 1
        r_squared = _dot(iterp.r, iterp.r)
        alpha = r_squared / dBd
        z_next = iterp.z + alpha * iterp.d

        # after 2
        r_next = iterp.r + alpha * Bd
        r_next_squared = _dot(r_next, r_next)

        # include a junk switch to catch the case where none should be executed
        index = jnp.argmax(
            jnp.array(
                [
                    False,
                    jnp.less_equal(dBd, 0),
                    jnp.greater_equal(
                        jnpla.vector_norm(z_next, ord=norm), trust_radius
                    ),
                    jnp.less_equal(jnp.sqrt(r_next_squared), tolerance),
                ]
            )
        )
        result = lax.switch(index, [noop, step1, step2, step3], (iterp, z_next))

        # update the state for the next iteration
        beta_next = r_next_squared / r_squared
        d_next = -r_next + beta_next * iterp.d
        z = z_next
        r = r_next
        d = d_next

        state = _CGSteihaugState(
            z=z,
            r=r,
            d=d,
            step=result.step,
            hits_boundary=result.hits_boundary,
            converged=result.converged,
        )
        return state

    def cond_f(iterp: _CGSteihaugState) -> bool:
        return jnp.logical_not(iterp.converged)

    # perform inner optimization to solve the constrained
    # quadratic subproblem using cg
    result = lax.while_loop(cond_f, body_f, init_param)

    result_out = _QuadSubproblemResult(
        step=result.step,
        hits_boundary=result.hits_boundary,
        pred_f=soa(result.step),
        nfev=0,
        ngev=0,
        nhev=0,
        success=True,
    )

    return result_out
