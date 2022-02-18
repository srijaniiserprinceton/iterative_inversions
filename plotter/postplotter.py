import matplotlib.pyplot as plt
import numpy as np

datadir = './input_data_files'
plotdir = './plots'

def postplot(c_arr_true, c_arr_init, c_arr_fit):
    # the bspline basis
    bsp_basis = np.load(f'{datadir}/bsp_basis.npy')
    # the grid 
    x = np.load(f'{datadir}/x.npy')
    # true_profile
    y_true = np.load(f'{datadir}/y_clean.npy')
    # noisy profile
    y_noisy = np.load(f'{datadir}/y_noisy.npy')
    # initial profile
    y_init = c_arr_init @ bsp_basis
    # fit profile
    y_fit = c_arr_fit @ bsp_basis
    
    
    # plotting
    plt.plot(x, y_init, 'b', label='y_init')
    plt.plot(x, y_true, 'k', label='y_true')
    plt.plot(x, y_noisy, 'xb', label='y_noisy')
    plt.plot(x, y_fit, 'r', label='y_fit')

    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{plotdir}/fit_profile.png')
    
    
    
