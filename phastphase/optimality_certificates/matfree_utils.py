import jax
import jax.numpy as jnp
import jax.lax as lax
from typing import NamedTuple

def complex_lobpcg(A, prec,X0, num_eigenvectors, max_iters, tol):
    (C,eigvals) = rayleigh_ritz(X0,A)
    n_x = jnp.shape(X0)[1]
    Theta = jnp.diag(eigvals[0:n_x])
    
    #Handle first iteration explicitly to deal with empty P block
    X = X0@C
    R = A(X) - X@Theta
    W = prec(R)
    S = jnp.column_stack((X,W))
    (C, eigvals) = rayleigh_ritz(S,A )
    Theta = jnp.diag(eigvals[0:n_x])
    X_new = S@(C[:,0:n_x])
    
    R_new = A(X_new) - X_new@Theta
    
    P_new = (S[:,n_x:])@(C[n_x:,0:n_x])
    init_state = LOBPCG_STATE(X_new, R_new, P_new,eigvals[0:n_x],1)
    def cond_function(state): return lobpcg_not_converged(state, tol, num_eigenvectors, max_iters)

    def loop_body(state): return lobpcg_loop_body(A, prec, n_x, state)

    final_state = lax.while_loop(cond_function, loop_body, init_state)

    return (final_state.eigvals[0:num_eigenvectors], final_state.X[:,0:num_eigenvectors])

class LOBPCG_STATE(NamedTuple):
    X:jax.typing.ArrayLike
    R:jax.typing.ArrayLike
    P:jax.typing.ArrayLike
    eigvals:jax.typing.ArrayLike
    iter_number: int

def lobpcg_loop_body(A,prec,n_x, lobpcg_state:LOBPCG_STATE):
    X= lobpcg_state.X
    R = lobpcg_state.R
    P = lobpcg_state.P
    W = prec(R)
    S = jnp.column_stack((X, W, P))
    [C, eigvals] = rayleigh_ritz(S,A)
    Theta = jnp.diag(eigvals[0:n_x])
    X_new = S@(C[:,0:n_x])
    R_new = A(X_new) - X_new@Theta
    P_new = (S[:,n_x:])@(C[n_x:,0:n_x])
    return LOBPCG_STATE(X_new, R_new, P_new,eigvals[0:n_x], lobpcg_state.iter_number+1)

def rayleigh_ritz(V,A):
    VtV = jnp.conj(jnp.transpose(V))@V
    D = jnp.diag(1/jnp.sqrt(jnp.diag(VtV)))
    R = jax.scipy.linalg.cholesky(D@VtV@D)
    scaled_mat = jax.numpy.linalg.multi_dot([D,jnp.conj(jnp.transpose(V)),A(V),D])
    scaled_mat = jax.scipy.linalg.solve_triangular(R,scaled_mat,trans=2)    #Left Multiply by R^-H
    scaled_mat = jnp.transpose(jax.scipy.linalg.solve_triangular(R,jnp.transpose(scaled_mat),trans=1))
    (eigvals, eigvec_mat) = jax.scipy.linalg.eigh(scaled_mat)
    C = D@jax.scipy.linalg.solve_triangular(R, eigvec_mat)
    
    return (C, eigvals)

def lobpcg_not_converged(state:LOBPCG_STATE, tolerance, num_eigvecs, max_iters):
    x_norms = jnp.linalg.vector_norm(state.X[:,0:num_eigvecs], axis=0)
    desired_eigs = state.eigvals[0:num_eigvecs]
    r_norms = jnp.linalg.vector_norm(state.R[:,0:num_eigvecs], axis=0)

    normalzed_error = r_norms/(desired_eigs*x_norms)

    tolerance_not_reached = jnp.logical_not(jnp.all(jnp.less_equal(normalzed_error, tolerance)))
    below_max_iters = jnp.less(state.iter_number, max_iters)
    return jnp.logical_and(tolerance_not_reached,below_max_iters)



def lanczos(A, vector_size, num_iters, seed = 1337):
    '''
    Returns two tuple representing tridiagonal matrix of lanczos iteration. 
    
    '''  
    alphas=jnp.zeros(num_iters+1, dtype=jnp.complex128)
    betas = jnp.zeros(num_iters, dtype=jnp.complex128)
    key = jax.random.key(seed)
    v = jax.random.normal(key,(2, vector_size) )
    v = lax.complex(v[0,:],v[1,:])
    v1 = v/jnp.linalg.vector_norm(v)

    w1 = A(v1)
    alphas = alphas.at[0].set(jnp.vdot(w1,v1))
    w1 = w1 - alphas[0]*v1

    init_state = LANCZOS_STATE(w1,v1,alphas, betas)
    def loop_func(iter, iter_state): return lanczos_loop(A, iter, iter_state)

    output_state = lax.fori_loop(0, num_iters,loop_func, init_state )
    return (output_state.alphas, output_state.betas)


class LANCZOS_STATE(NamedTuple):
    w:jax.typing.ArrayLike
    v:jax.typing.ArrayLike
    alphas:jax.typing.ArrayLike
    betas:jax.typing.ArrayLike
def lanczos_loop(A, iter_number, iter_state:LANCZOS_STATE):
    w = iter_state.w
    v = iter_state.v
    beta = jnp.linalg.vector_norm(w)
    v_new = w/beta
    w_new = A(v_new)
    alpha = jnp.vdot(w_new, v_new)
    w_new = w_new - alpha*v_new - beta*v
    alphas_new = iter_state.alphas.at[iter_number+1].set(alpha) #alphas are one bigger than the betas
    betas_new = iter_state.betas.at[iter_number].set(beta)
    return LANCZOS_STATE(w_new, v_new, alphas_new, betas_new)



def min_inverse_eig_estimate(A, num_iters):
    '''
    Estimates the inverse of the minimum eigenvalue of A, for positive definite A.

    '''
    
    (c,lower) = jax.scipy.linalg.cho_factor(A)
    def Ainv(v): return jax.scipy.linalg.cho_solve((c,lower), v)
    vector_size = jnp.shape(A)[1]
    alphas, betas = lanczos(Ainv, vector_size, num_iters)
    return jnp.max(jax.scipy.linalg.eigh_tridiagonal(alphas, betas))