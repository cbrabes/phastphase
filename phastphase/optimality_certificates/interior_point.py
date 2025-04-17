'''
Implements the Coey-Kapelevich-Vielma (CKV) Modification of the Skajaa-Ye primal dual interior point method (https://arxiv.org/abs/2107.04262), 


We specialize the method for SOS Quadratic Progrms:

min_{x,t}   t:
        [t, h - x] in Q
        x in K_SOS(P)

Where P defines our SOS Basis. The dual is

max_z   -h'z_q:
        z_q = z_sos
        z0 = 1
        (z0, z_q) in Q
        z_sos in K*_{SOS}(P)
Where K* is the dual cone

We further assume that P is a sparse trigonometric basis

'''

import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import Callable, NamedTuple, Type, Union, Optional, Tuple
import math
from math import sqrt
import jax.scipy as jscipy
from jax import jvp, grad

class PrimalDualVariables(NamedTuple):
    '''
    Structure to store primal dual variables, equivalent
    to omega in CKV, with variables removed if they can be substituted out
    '''
    z:jax.typing.ArrayLike
    z_norm:float
    tau:float
    s_SOS:jax.typing.ArrayLike
    s_Q:jax.typing.ArrayLike
    s_norm:float
    kappa:float

class ProblemData(NamedTuple):
    y:jax.typing.ArrayLike
    mask:jax.typing.ArrayLike   #Mask used to define sparse trig basis

def initial_primal_dual_point(nearfield_nonzero_count, farfield_intensities):
    farfield_nonzero_count = jnp.size(farfield_intensities)
    L = nearfield_nonzero_count
    U = farfield_nonzero_count
    c_scaling = jnp.sqrt(L/2 + 2)
    z0 = jnp.full_like(farfield_intensities,jnp.sqrt(L/(2*U)) )
    s_Q0 = jnp.full_like(farfield_intensities,-jnp.sqrt(L/(2*U)) )
    s_norm0 = jnp.sqrt(L/2 + 2) 
    s_sos0 = jnp.full_like(farfield_intensities,2*jnp.sqrt(L/(2*U)) )
    x0 = s_sos0
    tau0=1
    kappa0=1
    z_norm0 = jnp.sqrt(L/2 + 2)*tau0