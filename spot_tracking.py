'''
 Copyright by Artem Vorontsov, Kaspersky Lab US, 2021
 email: artem7vorontsov@gmail.com
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from physics import init, sim, gen_traj
from perturbations import pert_gen, get_gain
from optimizers import gd, adam
from controller_img import reset_graph, ctrl_cell, ann_ctrl_tr, ann_ctrl_inf, get_nn_control
from video_writer import prepare_video_writer, to_video, delete_video_stream

# Physics -----------------------------------------------------
L = 10.0             # simulation area size (m)
N = 256              # window size
n_vars = 2           # number of optimization channels
target_type = 'rnd'  # target type - random motion
dt = 1.0             # sampling time in the interval (0.0, 1.0]

# Dynamics -----------------------------------------------------
# Simulation time
steps = 15000
train_steps = 10000

# Optimization -------------------------------------------------
# Perturbation strength
pert_pow = 1.0
delta = 0.01
deg = 1.0

# Parameters for the NN controller -----------------------------
batch_size = 1 #do not change!
window_size = 4

learning_rate_st = 0.01*1e-2
learning_rate_fin = 0.01*1e-2

alpha_reg_factor = 0.0
du_reg_factor = 0.01

filters = (15, 15, 15)
kernels = ([3, 3], [3, 3], [3, 3])

num_rnn_units = 20

# Output ------------------------------------------------------
# Writing of results
# To video file (1) or plotting tracking performance and trajectories (0)
video = 0
# Simulation steps per frame
fpsim = 15
filename = 'video.mp4'

# Auxiliary parameters -----------------------------------------
# Allow mixed precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
tf_DTYPE = tf.float32
np_DTYPE = np.float32

# GPU/CPU usage for graph computations
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

gpuflag = 1
if gpuflag == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
# Physics
# Initialization of 2d scene and performance metric
    al = 2.0
    be = 5.0
    hat, J0, X, Y = init(L, N, al, be)

# Computation of target trajectory
    tr = gen_traj(target_type, steps)

# Setting up video writer
    vcont = []
    if video:
        vcont = prepare_video_writer(steps, L, filename)

# ANN controller - init
    reset_graph()
    cell = ctrl_cell(n_vars=n_vars,
                     batch_size=batch_size, dim=N,
                     num_rnn_units=num_rnn_units, ker_type='gru', num_dense=1,
                     filters=filters, kernels=kernels)

    ctrl_inf = ann_ctrl_inf(cell=cell)
    ctrl_tr = ann_ctrl_tr(cell=cell, unroll_size=window_size,
                          du=delta, pert_pow=pert_pow,
                          alpha_reg_factor=alpha_reg_factor, du_reg_factor=du_reg_factor)

    pars = {'window_size': window_size,
            'dim': N,
            'n_vars': n_vars,
            'train_steps': train_steps,
            'learning_rate_st': learning_rate_st,
            'learning_rate_fin': learning_rate_fin}

# Initialization
    Jt = 1.0
    u = np.zeros((n_vars,), dtype=np_DTYPE)
    u1, u2 = np.zeros_like(u), np.zeros_like(u)

# Control loop
    loss, tr_gd, tr_nn = [], [], []
    state1, state2 = [], []
    t = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(0, steps):
# Synthesize control and generate perturbation
            pert = pert_gen(Jt, 'spgd', n_vars, pert_pow, delta, deg)
            gain = get_gain(Jt, 100.0, deg)
            v0 = u - 0.5*pert
            v1 = u + 0.5*pert

# Apply control and perturbations
            Jt, frame0, pos = sim(v0, tr, be, X, Y, hat, J0, t)
            J1, frame1, pos = sim(v1, tr, be, X, Y, hat, J0, t)
            dJt = J1 - Jt

# AI controller
            grad = dJt*pert
            # SPGD control for verification of gradient estimation or for the combined control (see https://arxiv.org/abs/2204.05227)
            # u1, state1 = gd(u1, grad*gain, state1, learning_rate=1)
            u2, state2 = get_nn_control(frame0, Jt, grad, u1, sess, ctrl_inf, ctrl_tr, pars, state2)
            u = u1 + u2

            loss.append(Jt)
            tr_gd.append(u1)
            tr_nn.append(u2)

            t = t + dt

            if video and not np.mod(j, fpsim):
                if j < train_steps:
                    label = 'Training'
                else:
                    label = 'Inference'

                to_video(pos, u, loss, label, vcont)
                print('Frame {0:d}/{1:d} ...'.format(int(j/fpsim) + 1, int(steps/fpsim)))

    tr_gd, tr_nn = np.stack(tr_gd, axis=0), np.stack(tr_nn, axis=0)

# Plotting results
    if video:
    # Delete video stream
        delete_video_stream(vcont)
    else:
        plt.figure(0)
        plt.plot(loss)

        for k in range(n_vars):
            plt.figure(k + 1)
            plt.plot(tr[k], 'b')
            plt.plot(tr_gd[:, k] + tr_nn[:, k])
            plt.plot(tr_nn[:, k])

        plt.show()

if __name__ == "__main__":

    main()
