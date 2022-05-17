'''
 Copyright by Artem Vorontsov, Kaspersky Lab US, 2021
 email: artem7vorontsov@gmail.com
'''

import numpy as np
from sklearn.preprocessing import StandardScaler

np_DTYPE = np.float32

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def init(L, N, al, be):
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    hat = np.exp(-al*(X**2 + Y**2))
    J0 = np.sum(hat*np.exp(-be*(X**2 + Y**2)))

    return hat, J0, X, Y
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sim(u, tr, be, X, Y, hat, J0, t, frame_noise_pow=0.0):

    xt, yt = tr[0], tr[1]
    z = np.random.randn(hat.shape[0], hat.shape[1])
    x, y = interp(xt, t), interp(yt, t)
    frame = np.exp(-be*((X - (x - u[0]))**2 + (Y - (y - u[1]))**2)) + frame_noise_pow*z
    J = metric(frame, hat, J0)

    return J, frame, [x, y]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def metric(frame, hat, J0):

    M = frame*hat
    J = 1.0 - np.sum(M)/J0

    return J
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def interp(xt, t):

    if t == np.floor(t):
        x = xt[int(t)]
    else:
        ind0 = int(np.floor(t))
        ind1 = int(min(np.ceil(t), len(xt)))
        x = ((ind1 - t)*xt[ind0] + (t - ind0)*xt[ind1])

    return x
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_traj(type, len, ker_type='exp', ker_size=2000, power=1, vel=500):

    if type == 'rnd':
        tr = gen_random_traj((len, 2), ker_type=ker_type, ker_size=ker_size, power=power)

    else:
        tr = [np.sin(np.asarray(range(len))/vel), np.cos(np.asarray(range(len))/vel)]

    return tr
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_random_traj(shape, ker_type, ker_size, power):

    (L, n_vars) = shape

    if L < ker_size:
        w = np.random.randn(ker_size, n_vars).astype(np_DTYPE)
    else:
        w = np.random.randn(L, n_vars).astype(np_DTYPE)

    if ker_type == 'exp/2':
        x = np.linspace(-3.0, 0.0, ker_size, dtype=np_DTYPE)
        ker = np.exp(-x**2)
    elif ker_type == 'exp':
        x = np.linspace(-3.0, 3.0, ker_size, dtype=np_DTYPE)
        ker = np.exp(-x**2)
    elif ker_type == 'haar':
        ker = np.ones((ker_size,), dtype=np_DTYPE)
    else:
        ker = np.ones((1,), dtype=np_DTYPE)

    res = convolve(w, ker, 'periodic')
    scaler = StandardScaler()
    tr = power*scaler.fit_transform(res)

    if L < ker_size:
        tr = tr[:L, :]

    return list(tr.T)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def convolve(w, ker, mode):
    L, n_vars = w.shape
    n_ker = len(ker)

    res = []
    if mode == 'periodic':
        ker_ = np.zeros((L,), dtype=np_DTYPE)
# Inverse order of ker for using fft(ker_) instead of ifft(ker_)
        ker_[:n_ker] = ker[n_ker - 1 - np.asarray(range(n_ker))]

        tmp1 = ker_/np.sum(ker)
        tmp2 = np.fft.fft(tmp1)
        tmp2 = np.stack([tmp2]*n_vars, axis=1)
        tmp3 = np.fft.fft(w, axis=0)
        tmp4 = np.fft.ifft(np.multiply(tmp2, tmp3), axis=0)
        res = np.real(tmp4)

    elif mode == 'smooth':
        w_ = np.zeros((L + 2*n_ker, n_vars), dtype=np_DTYPE)
        w_[n_ker : L + n_ker, :] = w
        ker_ = np.zeros((L + 2*n_ker,), dtype=np_DTYPE)
        ker_[:n_ker] = ker[n_ker - 1 - np.asarray(range(n_ker))]

        tmp1 = ker_/np.sum(ker)
        tmp2 = np.fft.fft(tmp1)
        tmp2 = np.stack([tmp2] * n_vars, axis=1)
        tmp3 = np.fft.fft(w_, axis=0)
        tmp4 = np.fft.ifft(np.multiply(tmp2, tmp3), axis=0)
        res = np.real(tmp4)
        res = res[n_ker : L + n_ker, :]

    else:

        print('Please, specify the convolution mode!')

    return res