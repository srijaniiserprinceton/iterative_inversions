import build_inversion_setup as inv_setup
from src_iterinvPy import iterative_newton

# initializing the class for carrying out a single Newton inversion 
invertor = iterative_newton.inv_Iterative(inv_setup.func_dict, inv_setup.inv_dicts)

# running one newton inversion 
invertor.iterative_Newton()
