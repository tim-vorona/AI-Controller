'''
 Copyright by Artem Vorontsov, Kaspersky Lab US, 2021
 email: artem7vorontsov@gmail.com
'''

import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gd(var, grad, state, learning_rate=1.0):

    var = var - learning_rate*grad

    return var, state
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def rms_prop(var, grad, state, learning_rate=1.0, decay=0.9, momentum=0.0, eps=1e-10, centered=False):

# Initialization
    if state == []:
        state = {'ms': np.zeros_like(grad),
                 'mom': 0.0}

    mstm1 = state['ms']
    momtm1 = state['mom']

# State update rule
    if centered:
        mean_grad = decay*mstm1 + (1.0 - decay)*grad
        mean_square = decay*mstm1 + (1.0 - decay)*np.linalg.norm(grad)**2
        mom = momentum*momtm1 + learning_rate*grad/np.sqrt(mean_square - np.linalg.norm(mean_grad)**2 + eps)

    else:
        mean_square = decay*mstm1 + (1.0 - decay)*np.linalg.norm(grad)**2
        mom = momentum*momtm1 + learning_rate*grad/np.sqrt(mean_square + eps)

# Gradient descent rule
    var = var - mom

    state['ms'] = mean_square
    state['mom'] = mom

    return var, state
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def momentum(var, grad, state, learning_rate=1.0, momentum=0.9):

# Initialization
    if state == []:
        state = np.zeros_like(grad)

# State update rule
    state = momentum*state + grad

# Gradient descent rule
    var = var - learning_rate*state

    return var, state
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def adam(var, grad, state, learning_rate=1.0, beta1=0.9, beta2=0.999, eps=1e-8):

# Initialization
    if state == []:
        state = {'t': 0,
                 'm': np.zeros_like(grad),
                 'v': 0.0}

    tm1 = state['t']
    mtm1 = state['m']
    vtm1 = state['v']

    t = tm1 + 1
    lrt = learning_rate*np.sqrt(1.0 - beta2**t)/(1.0 - beta1**t)

# State update rule
    mt = beta1*mtm1 + (1.0 - beta1)*grad
    vt = beta2*vtm1 + (1.0 - beta2)*np.linalg.norm(grad)**2

# Gradient descent rule
    var = var - lrt*mt/(np.sqrt(vt) + eps)

    state['t'] = t
    state['m'] = mt
    state['v'] = vt

    return var, state

