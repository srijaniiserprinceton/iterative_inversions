import numpy as np
import matplotlib.pyplot as plt

# directory to store the data
datadir = './input_data_files'
# directory to store the plots
plotdir = './plots'
#--------------------------------------------------#
# creating the grid
xmin, xmax, N = 0, 2*np.pi, 201
# specifying the Gaussian noise
mean, std = 0.0, 0.25
# the function
f = lambda x: np.sin(x) + 0.3 * np.sin(10.*x)
#--------------------------------------------------#

x = np.linspace(xmin, xmax, N)
y = f(x)
y_err = np.random.normal(loc=mean,
                         scale=std,
                         size=len(y))
C_d = np.diag(np.ones_like(y) * std**2)

#--------saving the grid and the data (with/without noise-------#
np.save(f'{datadir}/x.npy', x)
np.save(f'{datadir}/y_clean.npy', y)
np.save(f'{datadir}/y_noisy.npy', y+y_err)
np.save(f'{datadir}/C_d.npy', C_d)

# plotting the clean and the noisy data
plt.plot(x, y, 'k', label='y_clean')
plt.plot(x, y + y_err, 'xb', label='y_noisy')
plt.xlabel('x')
plt.ylabel('data')
plt.legend()
plt.tight_layout()
plt.savefig(f'{plotdir}/data.png')
