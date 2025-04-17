import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Callable, NamedTuple, Type, Union, Optional, Tuple
import math
from math import sqrt
import jax.scipy as jscipy
from jax import jvp, grad

def damped_step(barrier_parameter, decrement):
    return barrier_parameter*decrement

def intermediate_step(barrier_parameter, decrement):
    return ((barrier_parameter*decrement)**2)/(1 + barrier_parameter*decrement)

class NewtonState(NamedTuple):
    x: jax.typing.ArrayLike
    decrement: float
    iter: int

#We use convention that delta_x = -inv(H)*grad, and x_{k+1} = x_k + t delta_x
def cg_newton_step_decrement(hvp_func, gradient, preconditioner, cg_tolerance, max_iters):
    x_step = jscipy.sparse.linalg.cg(hvp_func, gradient,tol=cg_tolerance, maxiter=max_iters, M=preconditioner)[0]
    newton_decrement = jnp.sqrt(jnp.dot(x_step, gradient))
    return (-x_step, newton_decrement)

def damped_barrier_netwon_method(grad_func, 
                                 hvp_func, 
                                 x0, 
                                 preconditioner, 
                                 barrier_parameter, 
                                 cg_tolerance, 
                                 newton_tolerance, 
                                 max_cg_iters, 
                                 max_newton_iters):
    
    def hvp(v): return hvp_func(x0, v)
    def prec(v): return preconditioner(v, x0)
    delta_x, newton_decrement = cg_newton_step_decrement(hvp, grad_func(x0), prec,cg_tolerance, max_cg_iters)
    init_params = NewtonState(x0,newton_decrement, 1)
    
    def newton_body_func(iter_state):
        gradient = grad_func(iter_state.x)
        def hvp(v): return hvp_func(iter_state.x, v)
        def prec(v): return preconditioner(v, iter_state.x)
        (delta_x,decrement) = cg_newton_step_decrement(hvp, gradient, prec, cg_tolerance, max_cg_iters)
        damping = lax.cond(decrement >= 1/(2*barrier_parameter),damped_step, intermediate_step,barrier_parameter, decrement  )
        x_new = iter_state.x+(1/(1+damping))*delta_x
        new_iter = iter_state.iter+1
        new_state = NewtonState(x_new, decrement, new_iter)
        return new_state
    def cond_func(iter_state):
        return jnp.logical_and(jnp.greater(iter_state.decrement, newton_tolerance), jnp.less(iter_state.iter, max_newton_iters))
    

    result = lax.while_loop(cond_func, newton_body_func, init_params)
    return result.x



def id_func(x):
    return x

class BarrierState(NamedTuple):
    t: float
    x: jax.typing.ArrayLike


def barrier_method(cost_vec, 
                   barrier_func, 
                   barrier_parameter, 
                   x0,
                   total_tolerance = 1e-6,
                   mu =100.0,
                   preconditioner=id_func,
                   cg_tolerance = 1e-7, 
                   newton_tolerance = 1e-6, 
                   max_cg_iters = 300, 
                   max_newton_iters = 300):
    
    def hvp_func(x, v): return jvp(grad(barrier_func),(x,), (v,))[1]
    barrier_grad = grad(barrier_func)
    def grad_func(x, t): return t*cost_vec+barrier_grad(x)
    t0 = jnp.abs(jnp.dot(x0,cost_vec)) + 1
    num_outer_loops = jnp.astype(jnp.ceil(jnp.log(barrier_parameter/total_tolerance/t0)/jnp.log(mu)),int)
    
    def barrier_outer_body_func(iter, barrier_state:BarrierState):
        def grad_x(x): return grad_func(x, barrier_state.t)
        x_new = damped_barrier_netwon_method(grad_x, 
                                             hvp_func, 
                                             barrier_state.x,
                                             preconditioner,
                                             barrier_parameter,
                                             cg_tolerance,
                                             newton_tolerance,
                                             max_cg_iters,
                                             max_newton_iters)
        t_new = barrier_state.t*mu
        return BarrierState(t_new, x_new)
    result = lax.fori_loop(0, num_outer_loops, barrier_outer_body_func, BarrierState(t0, x0))

    return (result.x, jnp.dot(cost_vec,result.x))






