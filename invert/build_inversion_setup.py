import jax

import init_dict
import build_newton_components as newton_comps
from src_iterinvPy import make_dictionaries as make_dicts

#-----------------the user-defined functions-----------------#
model_fn = newton_comps.model_fn
data_misfit_fn = newton_comps.data_misfit_fn
model_misfit_fn = newton_comps.regularization_fn
loss_fn = newton_comps.loss_fn

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
func_dict['grad'] = grad_
func_dict['hess'] = hess_

#-------------retrieving the parameter dictionary-------------#
inv_dicts = make_dicts.make_dicts(init_dict.user_data_dict,
                                  init_dict.user_model_dict,
                                  init_dict.user_reg_dict,
                                  init_dict.user_loop_dict,
                                  init_dict.user_path_dict,
                                  init_dict.user_misc_dict)

