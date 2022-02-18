import jax.numpy as jnp
import jax

#-------------data misfit function-------------#
def model_fn(c_arr, G):
    return c_arr @ G

def data_misfit_fn(c_arr, data, G, C_d):
    data_res = data - c_arr @ G
    C_d_inv = jnp.linalg.inv(C_d)
    
    data_misfit = data_res @ C_d_inv @ data_res
    
    return data_misfit

#------------model misfit function-------------#
def model_misfit_fn(c_arr, D, mu):
    return mu * (c_arr @ D @ c_arr)

#---------------loss function-------------------#
def loss_fn(c_arr, data, G, C_d, D, mu):
    return data_misfit_fn(c_arr, data, G, C_d) +\
        model_misfit_fn(c_arr, D, mu)
    
def get_newton_components():
    # defining the gradient
    grad = jax.grad(loss_fn)
    
    # defining the hessian
    hess = jax.jacfwd(jax.jacrev(loss_fn))
    
    loss_fn_ = jax.jit(loss_fn)
    model_misfit_fn_ = jax.jit(model_misfit_fn)
    grad_ = jax.jit(grad)
    hess_ = jax.jit(hess)

    return loss_fn_, model_misfit_fn_, grad_, hess_
