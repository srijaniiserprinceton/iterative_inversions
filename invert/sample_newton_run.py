import build_inversion_setup as inv_setup
from src_iterinvPy import newton_stepper

# initializing the class for carrying out a single Newton inversion
invertor = newton_stepper.inv_Newton(inv_setup.func_dict, inv_setup.inv_dicts)

# running one newton inversion
invertor.run_newton(invertor.data_total, invertor.c_init)
