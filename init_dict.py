import numpy as np
from src_iterinvPy import make_dictionaries as make_dicts

datadir = './input_data_files'

# the data dictionary frontend

user_data_dict = {}
user_data_dict['data'] = np.load(f'{datadir}/D.npy')
user_data_dict['C_d'] = np.load(f'{datadir}/C_d.npy')

# the model dictionary frontend

user_model_dict = {}
user_model_dict['G'] = np.load(f'{datadir}/bsp_basis.npy')
user_model_dict['c_init'] = np.load(f'{datadir}/true_coeff_arr.npy') * 0.0

# the regularization dictionary frontend
user_reg_dict = {}
user_reg_dict['mu'] = 1e-3
user_reg_dict['D'] = np.load(f'{datadir}/D.npy')

# the loop dictionary frontend
user_loop_dict = {}
user_loop_dict['loss_threshold'] = 1e-11
user_loop_dict['maxiter'] = 20

# the path dictionary forntend
user_path_dict = {}
user_path_dict['outdir'] = '.'

# the miscellaneous field dictionary frontend
user_misc_dict = {}

# retrieving the final dictionary
inv_dicts = make_dicts.make_dicts(user_data_dict,
                                  user_model_dict,
                                  user_reg_dict,
                                  user_loop_dict,
                                  user_path_dict,
                                  user_misc_dict)
