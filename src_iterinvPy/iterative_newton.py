import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import init_dict
import newton_stepper
import build_newton_components as newton_comps

datadir = './input_data_files'
plotdir = './plots'

# miscellaneous functions from newton components                                             
loss_fn_, model_misfit_fn_, grad_fn_, hess_fn_ = newton_comps.get_newton_components()

def iterative_newton():
    # running the 2^0 iteration (the first step to get 0th order inversion)
    c_arr_0 = newton_stepper.run_newton(init_dict.inv_dicts)
    
    # plotting the first fit
    plt.plot(x, c_arr_0 @ bsp_basis, 'r', alpha=0.5, label='y_iter')

    # the data residual d1 = d - d0
    d0 = newton_comps.model_fn(c_arr_0, G)
    data_res = data - d0

    delta_k = 10

    # the c_arr which accumulates corrections
    c_arr = c_arr_0 * 1.0

    # the memory array which keeps track every 2^n iterations
    c_arr_mem = jnp.array([c_arr])
    
    k_iter = 1

    while(delta_k > 1e-2 and k_iter < 10):
        for i in range(2**k_iter):
            c_arr_corr =\
            newton_stepper.run_newton(LOOP_ARGS, c_arr_init, data_res, G, C_d, D, hess_inv)
            c_arr = jax.ops.index_add(c_arr, jnp.index_exp[:], c_arr_corr)
            
            # recalculating the data residual
            data_k = newton_comps.model_fn(c_arr, G)
            data_res = data - data_k

        c_arr_mem = jnp.append(c_arr_mem, jnp.array([c_arr]), axis=0)
        
        y_prev = c_arr_mem[-2] @ bsp_basis
        y_now = c_arr_mem[-1] @ bsp_basis
        delta_k_new = jnp.max((y_now - y_prev) / jnp.sqrt(jnp.diag(C_d)))

        print(f'[k = {k_iter}] delta_k = {delta_k_new}')
        
        k_iter += 1

        # if(delta_k_new > delta_k): break
        
        delta_k = delta_k_new

        # plotting subsequent fits
        plt.plot(x, c_arr @ bsp_basis, 'r', alpha=0.5)

    # the final step
    plt.plot(x, c_arr @ bsp_basis, '--g', label='y_final')

if __name__ == "__main__":
    # computing the hessian once (accurate for linear problem)                                
    hess_inv = jnp.linalg.inv(hess_fn_(c_arr_init, data, G, C_d, D, mu))
    
    #-----------------------Plotting----------------------------------#
    # loading the basis functions
    bsp_basis = np.load(f'{datadir}/bsp_basis.npy')
    # loading the grid
    x = np.load(f'{datadir}/x.npy')

    # setting the plotting
    plt.figure()
    plt.plot(x, data, 'xb', label='y_noisy')
    plt.plot(x, c_arr_true @ bsp_basis, 'k', label='y_true')
    plt.plot(x, c_arr_init @ bsp_basis, 'gray', label='y_init')
    
    iterative_newton()

    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.legend()
    plt.tight_layout()
    # saving the plot
    plt.savefig(f'{plotdir}/fit_iterative.png')
