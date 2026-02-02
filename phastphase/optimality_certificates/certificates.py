import jax
import jax.experimental
import jax.experimental.sparse
import jax.numpy as jnp
import jax.lax as lax
import jax.scipy.linalg as slinalg
from functools import partial
from .barrier_method import barrier_method
from .matfree_utils import complex_lobpcg, lanczos
from jax import grad, jvp
def lambda_linear_operator(x:jax.typing.ArrayLike, u:jax.typing.ArrayLike, indices,flat_indices, mask:jax.typing.ArrayLike):
    mask = mask.at[indices].set(x)
    far_field = jnp.fft.fft2(mask, norm='ortho')
    far_field = far_field*jnp.reshape(u,far_field.shape)
    near_field = jnp.fft.ifft2(far_field, norm='ortho')
    flat_near_field = jnp.ravel(near_field)
    return flat_near_field[flat_indices]


def create_lambda_matrix(u:jax.typing.ArrayLike,flat_indices,mask):
    u2d = jnp.reshape(u, jnp.shape(mask))
    convolution_vec = jnp.roll(jnp.flip(jnp.fft.ifft2(u2d)),(1,1),(0,1))
    return build_lambda(convolution_vec, flat_indices, flat_indices)


@partial(jnp.vectorize, excluded={0,1}, signature='()->(n)')
def build_lambda(convolution_vector, flat_indices_vector, flat_index):
    unravelled_index_tuple = jnp.unravel_index(flat_indices_vector, jnp.shape(convolution_vector))
    offset_tuple = jnp.unravel_index(flat_index, jnp.shape(convolution_vector))
    rolled_vec = (jnp.subtract(unravelled_index_tuple[0], offset_tuple[0]), jnp.subtract(unravelled_index_tuple[1], offset_tuple[1]))

    
    return convolution_vector[rolled_vec]

def log_det_barrier_function(u:jax.typing.ArrayLike, 
                jac_function):
    L = jac_function(u)
    (sign, logabsdet) = jnp.linalg.slogdet(L)
    return -logabsdet

def quadratic_barrier(u):
    return -1*jnp.log(1-jnp.square(jnp.linalg.vector_norm(u)))
def cholesky_logdet(A):
    L = jnp.linalg.cholesky(A)
    d = jnp.real(jnp.diag(L))
    return -2*jnp.sum(jnp.log(d))

def eig_logdet(A):
    eigvals = jnp.linalg.eigvalsh(A)
    return -jnp.sum(jnp.log(eigvals))
#to-add: non-zero size + number of mu multiply loops

#We construct the preconditioner through a series of functions
def preconditioner_invA(v,x, eig_max, omega_weight = 1/2):
    z = 1 -jnp.dot(x,x)
    alpha = 2 + z*eig_max*omega_weight
    output = v - x*(4 * jnp.dot(x,v))/(z*alpha+4*jnp.dot(x,x))

    return (z/alpha)*output

def preconditioner_M(v,x,eig_max,SOS_HVP, omega_weight=1/2):
    v_intermediate = preconditioner_invA(v, x, eig_max, omega_weight)
    B_output = SOS_HVP(v_intermediate)-eig_max*omega_weight*v_intermediate
    intermediate_1 = v - B_output
    return preconditioner_invA(intermediate_1, x, eig_max, omega_weight)

def simple_preconditioner(v, x,lambda_mat_func,sos_hvp_func, scaling_parameter):
    #eigs = jnp.linalg.eigvalsh(lambda_mat_func(x))
    #min_eig = jnp.min(eigs)
    #max_hess_eig = scaling_parameter/(min_eig**2)
    def sos_hvp(v): return sos_hvp_func(x,v)
    return preconditioner_M(v,x, 1, sos_hvp, omega_weight=0)

def initial_max_eig(L,U):
    return L/U/init_central_scaling(L,U)



def init_central_scaling(L, U):
    return jnp.sqrt(2*L/U /((3+L) + jnp.sqrt((3+L)**2 - 4*L)))

def update_eigenvectors(lambda_matrix, apply_lambda_func, eigenvectors,max_iters,tol):
    lambda_cho = slinalg.cho_factor(lambda_matrix)
    def prec(v):return slinalg.cho_solve(lambda_cho, v)
    (eigvals, new_eigenvectors) = complex_lobpcg(apply_lambda_func,prec,eigenvectors, jnp.shape(eigenvectors)[1],max_iters,tol )
    return (eigvals[0], new_eigenvectors)



def phastphase_certificate(far_field_intensities:jax.typing.ArrayLike,
                           mask:jax.typing.ArrayLike, non_zero_size,
                           total_tolerance = 1e-8,
                           mu =100.0,
                   cg_tolerance = 1e-9, 
                   newton_tolerance = 1e-9, 
                   max_cg_iters = 300, 
                   max_newton_iters = 500):

    flat_indices = jnp.flatnonzero(mask, size=non_zero_size)

    def jac_func(u): return create_lambda_matrix(u, flat_indices,mask)

    def barrier_func(u): return log_det_barrier_function(u, jac_func) +quadratic_barrier(u)
    def sos_barrier(u): return log_det_barrier_function(u, jac_func)
    cost_vec = jnp.ravel(far_field_intensities)
    
    
    barrier_parameter = 2 + non_zero_size
    def sos_hvp_func(x, v): return jvp(grad(sos_barrier),(x,), (v,))[1]
    L = non_zero_size
    U = jnp.size(far_field_intensities)
    u0 = jnp.ones_like(cost_vec)*init_central_scaling(L,U)
    #def preconditioner(v,x): return simple_preconditioner(v,x, jac_func, sos_hvp_func, L/U)
    def preconditioner(v,x): return v
    cost_estimate = -barrier_method(cost_vec, barrier_func, barrier_parameter, u0, preconditioner=preconditioner,
                                    total_tolerance = total_tolerance,
                           mu =mu,
                   cg_tolerance = cg_tolerance, 
                   newton_tolerance = newton_tolerance, 
                   max_cg_iters = max_cg_iters, 
                   max_newton_iters = max_newton_iters)[1]
    return jax.nn.relu(cost_estimate)

def test_func(far_field_intensities:jax.typing.ArrayLike,
                           mask:jax.typing.ArrayLike, non_zero_size):
    flat_indices = jnp.flatnonzero(mask, size=non_zero_size)

    def jac_func(u): return create_lambda_matrix(u, flat_indices,mask)

    def barrier_func(u): return log_det_barrier_function(u, jac_func) +quadratic_barrier(u)

    L = non_zero_size
    U = jnp.size(far_field_intensities)
    cost_vec = jnp.ravel(far_field_intensities)
    u0 = jnp.ones_like(cost_vec)*init_central_scaling(L,U)
    barrier_grad = jax.grad(barrier_func)

    hess_func = jax.hessian(barrier_func)
    hessian = hess_func(u0)
    z = 1 - jnp.dot(u0,u0)
    

    hess_Q = jax.hessian(quadratic_barrier)(u0)
    def SOS_Barrier(u):return log_det_barrier_function(u, jac_func)
    hess_SOS = jax.hessian(SOS_Barrier)(u0)

    eig_min_lam = jnp.min(jnp.linalg.eigvalsh(jac_func(u0)))
    lam_max = L/U/(eig_min_lam**2)
    A = hess_Q + .5*lam_max*jnp.eye(U)
    B = hess_SOS - .5*lam_max*jnp.eye(U)
    Ainv = jnp.linalg.inv(A)
    Prec = Ainv@(jnp.eye(U) - B@Ainv)
    
    def hvp_fun(x): return hess_SOS@x

    key = jax.random.key(42)
    key2 = jax.random.key(55)
    V0 = jax.random.normal(key, (L, 1)) + 1j*jax.random.normal(key2,(L,1))
    def sos_barrier(u): return log_det_barrier_function(u, jac_func)
    def sos_hvp_func(x, v): return jvp(grad(sos_barrier),(x,), (v,))[1]
    def preconditioner(v,x): return simple_preconditioner(v,x, jac_func, sos_hvp_func, L/U)
    #(theta, V, i) = jax.experimental.sparse.linalg.lobpcg_standard(-jac_func(u0), V0,m=4)
    def prec(v): return preconditioner(v,u0)
    hess_out = jnp.zeros_like(hessian)
    for k in range(U):
        hess_out = hess_out.at[:,k].set(prec(hessian[:,k]))
    return jnp.linalg.cond(hess_out)


