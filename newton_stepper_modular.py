import numpy as np
import time 
import jax.numpy as jnp
from jax import tree_multimap
from collections import namedtuple
import build_newton_components as newton_comps
import sys

sys.path.append("./plotter")
import postplotter

datadir = './input_data_files'

# miscellaneous functions from newton components
loss_fn_, model_misfit_fn_, grad_fn_, hess_fn_ = newton_comps.get_newton_components()

def print_info(itercount, tdiff, data_misfit, loss_diff, max_grads, model_misfit):
    print(f'[{itercount:3d} | ' +
          f'{tdiff:6.1f} sec ] ' +
          f'data_misfit = {data_misfit:12.5e} ' +
          f'loss-diff = {loss_diff:12.5e}; ' +
          f'max-grads = {max_grads:12.5e} ' + 
          f'model_misfit={model_misfit:12.5e}')
    return None

def update(c_arr, grads, hess_inv):
    return tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)

def run_newton(LOOP_ARGS, c_arr, data, G, C_d, D, hess_inv):
    itercount = 0
    loss_diff = 1e25
    loss = LOOP_ARGS.loss
    
    while ((abs(loss_diff) > LOOP_ARGS.loss_threshold) and
           (itercount < LOOP_ARGS.maxiter)):
        t1 = time.time()
        
        #-----------------------------------------_---#
        loss_prev = loss
        grads = grad_fn_(c_arr, data, G, C_d, D, LOOP_ARGS.mu)
        c_arr = update(c_arr, grads, hess_inv)
        loss = loss_fn_(c_arr, data, G, C_d, D, LOOP_ARGS.mu)
        
        '''
        if loss > loss_prev:
            steplen /= 0.5
        '''
        
        model_misfit = model_misfit_fn_(c_arr, D, LOOP_ARGS.mu)
        data_misfit = loss -  model_misfit
        
        loss_diff = loss_prev - loss

        itercount += 1
        #---------------------------------------------#
        
        t2 = time.time()
        print_info(itercount, t2-t1, data_misfit, loss_diff, abs(grads).max(), model_misfit)

    return c_arr

if __name__ == "__main__":    
    # getting the various arguments for gradient and hessian
    c_arr_true = np.load(f'{datadir}/true_coeff_arr.npy')
    c_arr_init = np.zeros_like(c_arr_true)
    G = np.load(f'{datadir}/bsp_basis.npy')
    # data = np.load(f'{datadir}/y_synth.npy')
    data = np.load(f'{datadir}/y_noisy.npy')
    C_d = np.load(f'{datadir}/C_d.npy')
    D = np.load(f'{datadir}/D.npy')
    mu = 1e-3
    
    # the initial loss
    loss = loss_fn_(c_arr_init, data, G, C_d, D, mu)

    # arguments for the loop of newton stepper
    loop_args = namedtuple('LOOP_ARGS',
                           ['loss_threshold','maxiter', 'loss', 'mu'])
    LOOP_ARGS = loop_args(1e-12, 20, loss, mu)

    # creating the dictionary for data components
    userdict_data = {}
    userdict_data['data'] = data
    userdict_data['C_d'] = C_d

    # creating the dictionary for model function
    userdict_model_fn = {}
    userdict_model_fn['G'] = G
    userdict_model_fn['F'] = np.zeros_like(data)

    # creating the dictionary for regularization
    userdict_reg = {}
    userdict_reg['D'] = D
    userdict_reg['mu'] = mu

    # creating the dictionary of paths to be used to store files
    userdict_paths = {}
    userdict_paths['datadir'] = './input_data_files'

    # computing the hessian once (accurate for linear problem)
    hess_inv = jnp.linalg.inv(hess_fn_(c_arr_init, data, G, C_d, D, mu))
    c_arr_newton = run_newton(LOOP_ARGS, c_arr_init, data, G, C_d, D, hess_inv)

    print(c_arr_newton/c_arr_true)

    # plotting the fitted curve
    postplotter.postplot(c_arr_true, c_arr_init, c_arr_newton)
