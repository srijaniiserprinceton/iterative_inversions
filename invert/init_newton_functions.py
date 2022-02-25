import jax.numpy as jnp

#-------------------model misfit function--------------------#
def model_fn(c_arr, G):
    return c_arr @ G


#-------------------data misfit function---------------------#
def data_misfit_fn(c_arr, data, G, C_d):
    data_res = data - model_fn(c_arr, G)
    C_d_inv = jnp.linalg.inv(C_d)
    data_misfit = data_res @ C_d_inv @ data_res
    
    return data_misfit


#------------------regularization function-------------------#
def regularization_fn(c_arr, D, mu):
    return mu * (c_arr @ D @ c_arr)


#---------------------loss function--------------------------#
def loss_fn(c_arr, data, G, C_d, D, mu):
    return data_misfit_fn(c_arr, data, G, C_d) +\
        regularization_fn(c_arr, D, mu)

