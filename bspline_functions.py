import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from avni.tools.bases import eval_splrem, eval_polynomial, eval_vbspl

NAX = np.newaxis

# data directory
datadir = './input_data_files'
# plot directory
plotdir = './plots'

def create_knots_and_bsp(x, knot_num):
    xmin, xmax = x.min(), x.max()
    total_knot_num = knot_num
                     
    num_skip = len(x)//total_knot_num
    knot_locs = x[::num_skip]
    
    # st line basis at the ends and cubic B-splines in the center
    vercof1, dvercof1 = eval_polynomial(x, [xmin, xmax],
                                        1, types= ['TOP','BOTTOM'])
    vercof2, dvercof2 = eval_splrem(x, [xmin, xmax], knot_num)

    # arranging the basis from left to right with st lines                                
    bsp_basis = np.column_stack((vercof1[:, -1],
                                 vercof2[:, 1:-1],
                                 vercof1[:, 0]))
    d_bsp_basis = np.column_stack((dvercof1[:, -1],
                                   dvercof2[:, 1:-1],
                                   dvercof1[:, 0]))

    # making them of shape (n_basis, r)                                                   
    bsp_basis = bsp_basis.T
    d_bsp_basis = d_bsp_basis.T

    # storing the analytically derived B-splines and it first derivatives                 
    np.save(f'{datadir}/bsp_basis.npy', bsp_basis)
    
    # saving the plot for the basis functions
    for i in range(len(bsp_basis)):
        plt.plot(x, bsp_basis[i])
        
    plt.savefig(f'{plotdir}/bsp_basis.png')
    plt.close()

def find_bspline_coeffs(y):
    bsp_basis = np.load(f'{datadir}/bsp_basis.npy')
    
    # creating the carr corresponding to the DPT using custom knots                        
    Gtg = bsp_basis @ bsp_basis.T   # shape(n_basis, n_basis)
    
    # finding the coefficient array
    coeff_arr = np.linalg.inv(Gtg) @ (bsp_basis @ y)

    # saving the true coeff array
    np.save(f'{datadir}/true_coeff_arr.npy', coeff_arr)

    # plotting the reconstructed profile
    y_recon = coeff_arr @ bsp_basis
    np.save(f'{datadir}/y_synth.npy', y_recon)
    
    plt.plot(x, y, 'k', label='y_clean')
    plt.plot(x, y_recon, '--r', label='y_recon')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plotdir}/y_recon.png')
    plt.close()

def get_D_matrix():
    bsp_basis = np.load(f'{datadir}/bsp_basis.npy')
    x = np.load(f'{datadir}/x.npy')
    
    num_bsp = bsp_basis.shape[0]

    # shape (num_bsp x len_x)
    d2r_dbsp2 = np.zeros_like(bsp_basis)
    
    for i in range(num_bsp):
        dr_dbsp = np.gradient(bsp_basis[i], x, edge_order=2)
        d2r_dbsp2[i] = np.gradient(dr_dbsp, x, edge_order=2)

    D_bsp_j_D_bsp_k_r = d2r_dbsp2[:, NAX, :] * d2r_dbsp2[NAX, :, :]
    D = integrate.trapz(D_bsp_j_D_bsp_k_r, x, axis=2)
    
    #plotting and saving D matrix
    plt.pcolormesh(D)
    plt.colorbar()
    plt.tight_layout()
    
    plt.savefig(f'{plotdir}/D.png')
    np.save(f'{datadir}/D.npy', D)

if __name__ == "__main__":
    x = np.load(f'{datadir}/x.npy')
    knot_num = 40
    
    create_knots_and_bsp(x, knot_num)
    
    y = np.load(f'{datadir}/y_clean.npy')
    find_bspline_coeffs(y)

    get_D_matrix()
