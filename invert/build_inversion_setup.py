import jax

import init_dict
import init_newton_functions as newton_funcs
from src_iterinvPy import make_dictionaries as make_dicts

#-----------------the user-defined functions-----------------#
model_fn = newton_funcs.model_fn
data_misfit_fn = newton_funcs.data_misfit_fn
model_misfit_fn = newton_funcs.regularization_fn
loss_fn = newton_funcs.loss_fn

#---------------------gradient and hessian-------------------#                                
grad = jax.grad(loss_fn)
hess = jax.jacfwd(jax.jacrev(loss_fn))

#-------------------jitting the functions--------------------#                                
loss_fn_ = jax.jit(loss_fn)
model_misfit_fn_ = jax.jit(model_misfit_fn)
grad_ = jax.jit(grad)
hess_ = jax.jit(hess)

#------------------making the function dictionary-------------#
func_dict = {}
func_dict['loss_fn'] = loss_fn_
func_dict['model_misfit_fn'] = model_misfit_fn_
func_dict['grad_fn'] = grad_
func_dict['hess_fn'] = hess_

#-------------retrieving the parameter dictionary-------------#
inv_dicts = make_dicts.make_dicts(init_dict.user_data_dict,
                                  init_dict.user_model_dict,
                                  init_dict.user_reg_dict,
                                  init_dict.user_loop_dict,
                                  init_dict.user_path_dict,
                                  init_dict.user_misc_dict)

