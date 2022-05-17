'''
 Copyright by Artem Vorontsov, Kaspersky Lab US, 2021
 email: artem7vorontsov@gmail.com
'''

import numpy as np

np_DTYPE = np.float32

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def pert_gen(J, type, n_vars, pert_pow, du, deg):

    if type == 'uniform':
        pert = pert_pow*(J**deg)*du*np.sqrt(12)*(np.random.rand(n_vars).astype(np_DTYPE) - 0.5)
        std = pert_pow*(J**deg)*du

    elif type == 'normal':
        pert = pert_pow*(J**deg)*du*np.random.randn(n_vars).astype(np_DTYPE)
        std = pert_pow*(J**deg)*du

    elif type == 'spgd':
        amin = 0.01
        pert = (amin + pert_pow*(J**deg))*du*np.sqrt(12)*(np.random.rand(n_vars).astype(np_DTYPE) - 0.5)
        std = (amin + pert_pow*(J**deg))*du

    else:
        pert = np.zeros((n_vars,)).astype(np_DTYPE)
        std = 0

    return pert
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_gain(J, gain_pow, deg):
    amin = 0.0
    amax = gain_pow
    gain = amin + (amax - amin)*J**deg

    return gain
