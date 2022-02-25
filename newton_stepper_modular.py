import numpy as np
import time 
import jax.numpy as jnp
from jax import tree_multimap
from collections import namedtuple
import build_newton_components as newton_comps
import sys

sys.path.append("./plotter")
import postplotter


# miscellaneous functions from newton components
loss_fn_, model_misfit_fn_, grad_fn_, hess_fn_ = newton_comps.get_newton_components()

class inv_Newton:
    def __init__(self, inv_dicts):
        #--------unwraps the inversion dictionary (static parts only)-----#
        # the data components
        self.C_d = inv_dicts.data_dict['C_d']
        
        # the model components
        self.G = inv_dicts.model_dict['G']
    
        # the regularization components
        self.mu = inv_dicts.reg_dict['mu']
        self.D = inv_dicts.reg_dict['D']
        
        # the loop components
        self.loss_threshold = inv_dicts.loop_dict['loss_threshold']
        self.maxiter = inv_dicts.loop_dict['maxiter']
        
        # the miscellaneous components
        self.hessinv = inv_dicts.misc_dict['hessinv']

        # if the hessian inverse has not be precomputed yet
        self.hessinv = self.compute_hessinv()
        

    def print_info(self, itercount, tdiff, data_misfit,
                   loss_diff, max_grads, model_misfit):
        print(f'[{itercount:3d} | ' +
              f'{tdiff:6.1f} sec ] ' +
              f'data_misfit = {data_misfit:12.5e} ' +
              f'loss-diff = {loss_diff:12.5e}; ' +
              f'max-grads = {max_grads:12.5e} ' + 
              f'model_misfit={model_misfit:12.5e}')
        
        return None


    def update(self, c_arr, grads, hess_inv):
        return tree_multimap(lambda c, g, h: c - g @ h, c_arr, grads, hess_inv)


    def run_newton(self, data, c_arr):
        itercount = 0
        loss_diff = 1e25
        loss = LOOP_ARGS.loss
        
        while ((abs(loss_diff) > self.loss_threshold) and
               (itercount < self.maxiter)):
            
            # start time for an iteration
            t1 = time.time()
            
            #--------------Body of a newton step---------------#
            loss_prev = loss
            
            grads = grad_fn_(c_arr, data, self.G, self.C_d, self.D, self.mu)
            c_arr = update(c_arr, grads, self.hess_inv)
            loss = loss_fn_(c_arr, data, self.G, self.C_d, self.D, self.mu)
            
            model_misfit = model_misfit_fn_(c_arr, self.D, self.mu)
            data_misfit = loss -  model_misfit
            
            loss_diff = loss_prev - loss
            
            itercount += 1
            #---------------------------------------------#
            
            # end time for an iteration
            t2 = time.time()
            
            print_info(itercount, t2-t1, data_misfit,
                       loss_diff, abs(grads).max(), model_misfit)
            
        return c_arr
