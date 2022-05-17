'''
 Copyright by Artem Vorontsov, Kaspersky Lab US, 2021
 email: artem7vorontsov@gmail.com
'''

import tensorflow as tf
import numpy as np

tf_DTYPE = tf.float32
np_DTYPE = np.float32

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#AI-CONTROLLER FOR THE IMAGE INPUT
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def reset_graph():
    global sess
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def lr(t, Jt, learning_rate_st, learning_rate_fin):
# Constant hard
    # learning_rate = learning_rate_st
# Constant soft
    # learning_rate = learning_rate_fin
# Linear fading
    # learning_rate = learning_rate_st*(1.0 - t/time) + learning_rate_fin*(t/time)
# Metric-dependent
    learning_rate = learning_rate_st*Jt + learning_rate_fin*(1.0 - Jt)
# Exponental fading
    # learning_rate = learning_rate_st*np.exp(-4.0*t/time) + learning_rate_fin*(1.0 - np.exp(-4.0*t/time))
    return learning_rate
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ctrl_cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, n_vars=2,
                 batch_size=1, dim=256,
                 num_rnn_units=10, ker_type='gru', num_dense=1,
                 filters=(15, 15, 15), kernels=([3, 3], [3, 3], [3, 3]),
                 **kwargs):
        self._out_size = n_vars
        self._batch_size = batch_size
        self._dim = dim

        self._filters = filters
        self._kernels = kernels

        self._num_rnn_units = num_rnn_units
        self._ker_type = ker_type

        self._num_dense = num_dense

        self._initializer = tf.initializers.glorot_uniform()

        super(ctrl_cell, self).__init__(**kwargs)

    @property
    def state_size(self):
        if self._ker_type == 'lstm':
            return (2*self._num_rnn_units, self._out_size)
        else:
            return (1*self._num_rnn_units, self._out_size)

    @property
    def output_size(self):
        return (self._out_size, self._out_size)

    def __call__(self, inputs, state, scope=None):
        frame, J, mode, w = inputs[0], inputs[1], inputs[2], inputs[3]

        state_rnn_tm1 = state[0]
        utm1 = state[1]

        mode = tf.reduce_all(mode)

        frame_rsh = tf.reshape(frame, [self._batch_size, self._dim, self._dim, 1])

        initializer = self._initializer

        cnn_out = frame_rsh

# 2D CNN for pre-processing of image frame
        for j in range(len(self._filters)):
            shape = cnn_out.get_shape()
            if shape[1].value > self._kernels[j][0] and shape[2].value > self._kernels[j][1]:
                conv_output = tf.layers.conv2d(inputs=cnn_out,
                                               filters=self._filters[j],
                                               kernel_size=self._kernels[j],
                                               data_format='channels_last',
                                               name='conv' + str(j),
                                               reuse=tf.AUTO_REUSE,
                                               kernel_initializer=initializer)
                # bn_output = tf.layers.batch_normalization(inputs=conv_output,
                #                                           axis=3,
                #                                           reuse=tf.AUTO_REUSE,
                #                                           name='bn' + str(j),
                #                                           training=mode)
                bn_output = conv_output
                cnn_out = tf.layers.MaxPooling2D(pool_size=self._kernels[j],
                                                 strides=self._kernels[j],
                                                 data_format='channels_last')(bn_output)

        cnn_output = tf.layers.flatten(cnn_out)

        rnn_input = tf.concat([cnn_output, w, utm1], axis=1)

# Temporal processing of feature vector
        if self._ker_type == 'lstm':
            rnn_output, state_rnn_t = tf.nn.rnn_cell.LSTMCell(num_units=self._num_rnn_units,
                                                              name='rnn1',
                                                              use_peepholes=True,
                                                              reuse=tf.AUTO_REUSE,
                                                              initializer=initializer,
                                                              state_is_tuple=False)(inputs=rnn_input, state=state_rnn_tm1)
        elif self._ker_type == 'gru':
            rnn_output, state_rnn_t = tf.nn.rnn_cell.GRUCell(num_units=self._num_rnn_units,
                                                             name='rnn1',
                                                             reuse=tf.AUTO_REUSE,
                                                             kernel_initializer=initializer)(inputs=rnn_input,
                                                                                             state=state_rnn_tm1)
        elif self._ker_type == 'dense':
            rnn_output = tf.layers.dense(inputs=rnn_input,
                                         units=3*self._num_rnn_units,
                                         activation='tanh',
                                         name='rnn1',
                                         reuse=tf.AUTO_REUSE,
                                         kernel_initializer=initializer)
            state_rnn_t = state_rnn_tm1

        else:
            rnn_output, state_rnn_t = rnn_input, state_rnn_tm1

# Dense layers
        dns = rnn_output
        for j in range(self._num_dense):
            dns = tf.layers.dense(inputs=dns,
                                  units=10*self._out_size,
                                  activation='relu',
                                  name='dense' + str(j),
                                  reuse=tf.AUTO_REUSE,
                                  kernel_initializer=initializer)
# Output layer
        nn_output = tf.layers.dense(inputs=dns,
                                    units=self._out_size,
                                    activation='linear',
                                    name='out_dense',
                                    reuse=tf.AUTO_REUSE,
                                    kernel_initializer=initializer)

        new_state = [state_rnn_t, nn_output]
        output = [nn_output, utm1]

        return output, new_state
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ann_ctrl_inf(cell=None, scope='ann_controller'):
    dim = cell._dim
    n_vars = cell._out_size
    batch_size = cell._batch_size

# Graph for inference mode
    frame = tf.placeholder(dtype=tf_DTYPE, shape=(dim, dim), name='frame_inf')
    frame_in = tf.reshape(frame, [batch_size, 1, dim, dim])
    J = tf.placeholder(dtype=tf_DTYPE, shape=(), name='J_inf')
    J_in = tf.reshape(J, [batch_size, 1, 1])
    w = tf.placeholder(dtype=tf_DTYPE, shape=(n_vars,), name='w_inf')
    w_in = tf.reshape(w, [batch_size, 1, n_vars])
    mode = tf.get_variable(name='mode_inf',
                           initializer=lambda: tf.zeros((batch_size, 1, 1), dtype=tf.bool),
                           dtype=tf.bool,
                           trainable=False)

    states = []
    cnt = 0
    for size in cell.state_size:
        states.append(tf.get_variable(name='ctrl_state_' + str(cnt) + '_inf',
                                      initializer=lambda: tf.zeros((batch_size, size)),
                                      dtype=tf_DTYPE,
                                      trainable=False))
        cnt = cnt + 1

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        u, final_states = tf.nn.dynamic_rnn(cell, [frame_in, J_in, mode, w_in], initial_state=states)

    states_assing = []
    for j in range(len(states)):
        states_assing.append(tf.assign(states[j], final_states[j]))

    N_tr = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope=scope)])
    print("Number of all trainable variables in the ANN controller: ", N_tr)

    out = dict(frame=frame, w=w, J=J,
               states_assign=states_assing,
               u=u[0])
    return out
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ann_ctrl_tr(cell=None, scope='ann_controller', unroll_size=8, du=0.01, pert_pow=1.0, alpha_reg_factor=0.0, du_reg_factor=0.0):

    dim = cell._dim
    n_vars = cell._out_size
    batch_size = cell._batch_size

# Graph for training mode
    learning_rate = tf.placeholder(dtype=tf_DTYPE, shape=(), name='leaning_rate')
    frame = tf.placeholder(dtype=tf_DTYPE, shape=(batch_size, unroll_size, dim, dim), name='frame_tr')
    J = tf.placeholder(dtype=tf_DTYPE, shape=(batch_size, unroll_size, 1), name='J_tr')
    w = tf.placeholder(dtype=tf_DTYPE, shape=(batch_size, unroll_size, n_vars), name='w_tr')
    mode = tf.get_variable(name='mode_tr',
                           initializer=lambda: tf.ones((batch_size, unroll_size, 1), dtype=tf.bool),
                           dtype=tf.bool,
                           trainable=False)

    states = []
    cnt = 0
    for size in cell.state_size:
        states.append(tf.get_variable(name='ctrl_state_' + str(cnt) + '_tr',
                                      initializer=lambda: tf.zeros((batch_size, size)),
                                      dtype=tf_DTYPE,
                                      trainable=False))
        cnt = cnt + 1

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        u, final_states = tf.nn.dynamic_rnn(cell, [frame, J, mode, w], initial_state=states)

# Estimation of the gradient given from external sources
        grad = tf.placeholder(dtype=tf_DTYPE, shape=(batch_size, unroll_size, n_vars), name='grad_tr')

# Derivatives
        sum_over_ctrl = (1.0/n_vars)*tf.math.reduce_sum(grad*u[0], axis=2)
        sum_over_time = (1.0/unroll_size)*tf.math.reduce_sum(sum_over_ctrl, axis=1)
        sum_over_batch_dim = (1.0/(pert_pow*du)**2)*tf.math.reduce_sum(sum_over_time, axis=0)

# Regularization - smoothness of the output control
        u2 = (1.0/n_vars)*tf.math.reduce_sum((u[0] - u[1])*(u[0] - u[1]), axis=2)
        u2 = (1.0/unroll_size)*tf.math.reduce_sum(u2, axis=1)
        u2 = (1.0/(pert_pow*du))*tf.math.reduce_sum(u2, axis=0)

        tr_vars = tf.trainable_variables(scope=scope)
        loss = (1.0 - du_reg_factor)*sum_over_batch_dim + du_reg_factor*u2

        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1e-6)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, beta1=0.90, beta2=0.999, epsilon=1e-2
        # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        symb_grad = tf.gradients(loss, tr_vars)

# Regularization on the NN weights
        if alpha_reg_factor > 0:
            reg = []
            weights = tr_vars
            for weight in weights:
                reg.append(alpha_reg_factor*weight)

            symb_grad = symb_grad + reg

        grads_and_vars = list(zip(symb_grad, tr_vars))
        train_step = optimizer.apply_gradients(grads_and_vars)

    states_assing = []
    for j in range(len(states)):
        states_assing.append(tf.assign(states[j], final_states[j]))

    N_tr = np.sum([np.prod(v.get_shape().as_list()) for v in tr_vars])
    print("Number of all trainable variables in the scope '{0:s}': {1:d}".format(scope, N_tr))

    out = dict(frame=frame, w=w, J=J, grad=grad,
               states_assign=states_assing,
               learning_rate=learning_rate,
               u=u[0],
               train_step=train_step)
    return out
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_nn_control(framet, Jt, gradt, wt, sess, inf_graph, tr_graph, pars, state):

    batch_size = 1
    window_size = pars['window_size']
    dim = pars['dim']
    n_vars = pars['n_vars']
    train_steps = pars['train_steps']
    learning_rate_st = pars['learning_rate_st']
    learning_rate_fin = pars['learning_rate_fin']

# Initialization
    if len(state) == 0:
        state = {'frame_tr': np.zeros((batch_size, window_size, dim, dim), dtype=np_DTYPE),
                 'J_tr': np.zeros((batch_size, window_size, 1), dtype=np_DTYPE),
                 'grad_tr': np.zeros((batch_size, window_size, n_vars), dtype=np_DTYPE),
                 'w_tr': np.zeros((batch_size, window_size, n_vars), dtype=np_DTYPE),
                 't': 0}

    t = state['t']
    cnt = np.mod(t, window_size)

# Synthesize control
    if t > window_size:
        feed_dict_inf = {inf_graph['frame']: framet, inf_graph['J']: Jt, inf_graph['w']: wt}
        ut, _ = sess.run([inf_graph['u'], inf_graph['states_assign']], feed_dict_inf)
        u = ut.reshape(-1)

    else:
        u = np.zeros((n_vars,), dtype=np_DTYPE)

# Collect training data
    state['frame_tr'][0, cnt, :, :] = framet
    state['J_tr'][0, cnt, 0] = Jt
    state['grad_tr'][0, cnt, :] = gradt
    state['w_tr'][0, cnt, :] = wt

# Train controller
    if t > window_size and cnt == window_size - 1:
        learning_rate = lr(t, Jt, learning_rate_st, learning_rate_fin)
        feed_dict_tr = {tr_graph['frame']: state['frame_tr'], tr_graph['J']: state['J_tr'], tr_graph['w']: state['w_tr'],
                        tr_graph['grad']: state['grad_tr'],
                        tr_graph['learning_rate']: learning_rate}
        if t < train_steps:
            train_step, _ = sess.run([tr_graph['train_step'], tr_graph['states_assign']], feed_dict_tr)
        else:
            _ = sess.run(tr_graph['states_assign'], feed_dict_tr)

    state['t'] = t + 1

    return u, state