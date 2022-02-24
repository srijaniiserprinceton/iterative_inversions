import numpy as np
from src_iterinvPy import make_dictionaries as make_dicts

# the data dictionary frontend

user_data_dict = {}
user_data_dict['data'] = np.array([10.])

# the model dictionary frontend

user_model_dict = {}
user_model_dict['G'] = np.array([100.])

# the regularization dictionary frontend
user_reg_dict = {}
user_reg_dict['mu'] = 1.0
user_reg_dict['D'] = np.array([1e5])

# the path dictionary forntend
user_path_dict = {}
# user_path_dict['outdir'] = '.'

# the miscellaneous field dictionary frontend
user_misc_dict = {}

# retrieving the final dictionary
inv_dicts = make_dicts.make_dicts(user_data_dict,
                                  user_model_dict,
                                  user_reg_dict,
                                  user_path_dict,
                                  user_misc_dict)
