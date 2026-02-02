import torchmin
import torch


def augmented_lagrange_support(near_field_0, support_mask, loss_function, support_tol=1e-4, grad_tol=1e-8, tr_max_iter=100, outer_max_iter=100):
    x0 = torch.view_as_real_copy(near_field_0)
    def constraint_func(x):
        return torch.linalg.vector_norm(support_mask*torch.view_as_complex(x))
    constraint_scaling = max(1.0,torch.max(torch.abs(torch.autograd.functional.vjp(constraint_func,x0)[1])))
    def scaled_constraint(x):
        return constraint_func(x)/constraint_scaling
    
    loss_scaling = max(1.0,torch.max(torch.abs(torch.autograd.functional.vjp(loss_function,x0)[1])))
    def scaled_loss(x):
        return loss_function(x)/loss_scaling
    
    def lagrangian(x, lam, rho):
        return scaled_loss(x) + rho/2*torch.square(scaled_constraint(x)+lam/rho)
    rho = 20*scaled_loss(x0)/torch.square(scaled_constraint(x0))
    lam = 0
    
    x_k = x0.detach().clone() 
    for _ in range(outer_max_iter):
        if scaled_constraint(x_k) < support_tol:
            break
        else:
            def loss_lam(x): return lagrangian(x, lam, rho)
            h_km1 = scaled_constraint(x_k)
            result = torchmin.trustregion._minimize_trust_ncg(
                loss_lam,
                x_k,
                gtol=grad_tol,
                max_iter = tr_max_iter,
                disp = 2
            )
        
            x_k = result.x
            lam += rho*scaled_constraint(x_k)
            if scaled_constraint(x_k) > .9*h_km1:
                rho *= 10
    return x_k




