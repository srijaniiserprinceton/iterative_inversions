import numpy as np

datadir = '../input_data_files'

# the data dictionary frontend

user_data_dict = {}
user_data_dict['data'] = np.load(f'{datadir}/y_noisy.npy')
user_data_dict['C_d'] = np.load(f'{datadir}/C_d.npy')

# the model dictionary frontend

user_model_dict = {}
user_model_dict['G'] = np.load(f'{datadir}/bsp_basis.npy')
user_model_dict['c_init'] = np.load(f'{datadir}/true_coeff_arr.npy') * 0.0
user_model_dict['c_ref'] = np.load(f'{datadir}/true_coeff_arr.npy')

# the regularization dictionary frontend
user_reg_dict = {}
user_reg_dict['mu'] = 1.
user_reg_dict['D'] = np.load(f'{datadir}/D.npy')

# the loop dictionary frontend
user_loop_dict = {}
user_loop_dict['loss_threshold'] = 1e-12
user_loop_dict['maxiter'] = 20
user_loop_dict['k_iter_max'] = 10

# the path dictionary forntend
user_path_dict = {}
user_path_dict['outdir'] = '.'
user_path_dict['plotdir'] = '../plots'

# the miscellaneous field dictionary frontend
user_misc_dict = {}
user_misc_dict['hessinv'] = None
