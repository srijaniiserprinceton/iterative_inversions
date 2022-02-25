import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src_iterinvPy import newton_stepper

class inv_Iterative:
    def __init__(self, func_dict, inv_dicts):
        #-----------unwrapping the components needed-----------------#
        self.k_iter_max = inv_dicts.loop_dict['k_iter_max']
        self.G = inv_dicts.model_dict['G']
        self.model_fn = func_dict['model_fn']
        self.plotdir = inv_dicts.path_dict['plotdir']
        self.c_arr_ref = inv_dicts.model_dict['c_ref']

        #--------------initializing the newton solver----------------#
        self.newton_iterator = newton_stepper.inv_Newton(func_dict, inv_dicts,
                                                         isIterative=True)

    def iterative_Newton(self):
        # running the 2^0 iteration (the first step to get 0th order inversion)
        data, c_init = self.newton_iterator.data_total, self.newton_iterator.c_init
        
        c_arr_0 = self.newton_iterator.run_newton(data, c_init)
        
        # plotting the first fit
        plt.plot(self.c_arr_ref @ self.G, 'k', label='y_true')
        plt.plot(c_arr_0 @ self.G, 'r', alpha=0.2, label='y_iter')

        # changing c_init tp zeros for the perturbative fitting
        c_init = c_init * 0.0
        
        # the data residual d1 = d - d0
        d0 = self.model_fn(c_arr_0, self.G)
        data_res = data - d0
        
        # the c_arr which accumulates corrections
        c_arr = c_arr_0 * 1.0
        
        # the memory array which keeps track every 2^n iterations
        c_arr_mem = jnp.array([c_arr])
        
        # the counter for the number of iteration of newton inversion
        k_iter = 1
        
        while(k_iter < self.k_iter_max+1):
            for i in range(2**k_iter):
                c_arr_corr = self.newton_iterator.run_newton(data_res, c_init)
                                 
                c_arr = jax.ops.index_add(c_arr, jnp.index_exp[:], c_arr_corr)
                
                # recalculating the data residual
                data_k = self.model_fn(c_arr, self.G)
                data_res = data - data_k
                
            c_arr_mem = jnp.append(c_arr_mem, jnp.array([c_arr]), axis=0)
        
            # plotting subsequent fits
            plt.plot(c_arr @ self.G, 'r', alpha=0.2)
            
            k_iter += 1
            print(f'\n-------------------{k_iter}------------------\n')    
        # plotting the final step
        plt.plot(c_arr @ self.G, '--g', label='y_final')
        
        #------------------------------------------------------#
        plt.xlabel('x')
        plt.ylabel('y')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.plotdir}/fit_iterative.png')
