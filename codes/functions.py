import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


def neural_network(x, neurons, is_training, dtype=tf.float32, decay=0.9):
    def batch_normalization(y):
        shape = y.get_shape().as_list()
        y = tf.reshape(y, [-1, shape[1] * shape[2]])
        beta = tf.compat.v1.get_variable(
            name='beta', shape=[shape[1] * shape[2]],
            dtype=dtype, initializer=tf.zeros_initializer()
        )
        gamma = tf.compat.v1.get_variable(
            name='gamma', shape=[shape[1] * shape[2]],
            dtype=dtype, initializer=tf.ones_initializer()
        )
        mv_mean = tf.compat.v1.get_variable(
            name='mv_mean', shape=[shape[1] * shape[2]],
            dtype=dtype, initializer=tf.zeros_initializer(),
            trainable=False
        )
        mv_var = tf.compat.v1.get_variable(
            name='mv_var', shape=[shape[1] * shape[2]],
            dtype=dtype, initializer=tf.ones_initializer(),
            trainable=False
        )
        mean, variance = tf.nn.moments(y, [0], name='moments')
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_mean, mean, decay,
                                  zero_debias=True)
        )
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_var, variance, decay,
                                  zero_debias=False)
        )
        mean, variance = tf.cond(is_training, lambda: (mean, variance),
                                 lambda: (mv_mean, mv_var))
        y = tf.nn.batch_normalization(y, mean, variance, beta, gamma, 1e-6)
        return tf.reshape(y, [-1, shape[1], shape[2]])

    def fc_layer(y, out_size, activation, is_single):
        shape = y.get_shape().as_list()
        w = tf.compat.v1.get_variable(
            name='weights',
            shape=[shape[2], shape[1], out_size],
            dtype=dtype,
            initializer=tf.initializers.glorot_uniform()
        )
        y = tf.transpose(tf.matmul(tf.transpose(y, [2, 0, 1]), w),
                         [1, 2, 0])
        if is_single:
            b = tf.compat.v1.get_variable(
                name='bias',
                shape=[out_size, shape[2]],
                dtype=dtype,
                initializer=tf.zeros_initializer()
            )
            return activation(y + b)
        return activation(batch_normalization(y))

    x = batch_normalization(x)
    for i in range(len(neurons)):
        with tf.compat.v1.variable_scope('layer_' + str(i)):
            x = fc_layer(x, neurons[i],
                         tf.nn.relu if i < len(neurons) - 1
                         else tf.nn.sigmoid, False)
    return x


def deep_optimal_stopping(x, t, n, g, neurons, batch_size, train_steps,
                          mc_runs, lr_boundaries, lr_values, beta1=0.9,
                          beta2=0.999, epsilon=1e-8, decay=0.9):
    is_training = tf.compat.v1.placeholder(tf.bool, [])
    p = g(t, x)
    nets = neural_network(tf.concat([x[:, :, :-1], p[:, :, :-1]], axis=1),
                          neurons, is_training, decay=decay)

    u_list = [nets[:, :, 0]]
    u_sum = u_list[-1]
    for k in range(1, n - 1):
        u_list.append(nets[:, :, k] * (1 - u_sum))
        u_sum += u_list[-1]

    u_list.append(1 - u_sum)
    u_stack = tf.concat(u_list, axis=1)
    p = tf.squeeze(p, axis=1)
    loss = tf.reduce_mean(tf.reduce_sum(-u_stack * p, axis=1))
    idx = tf.argmax(tf.cast(tf.cumsum(u_stack, axis=1) + u_stack >= 1,
                            dtype=tf.uint8),
                    axis=1, output_type=tf.int32)
    stopped_payoffs = tf.reduce_mean(
        tf.gather_nd(p, tf.stack([tf.range(0, batch_size, dtype=tf.int32),
                                  idx], axis=1))
    )

    global_step = tf.Variable(0)
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step,
                                                          lr_boundaries,
                                                          lr_values)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate,
                                                 beta1=beta1,
                                                 beta2=beta2,
                                                 epsilon=epsilon)
    update_ops = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.UPDATE_OPS
    )
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for _ in range(train_steps):
            sess.run(train_op, feed_dict={is_training: True})

        px_mean = 0
        for _ in range(mc_runs):
            px_mean += sess.run(stopped_payoffs,
                                feed_dict={is_training: False})

    return px_mean / mc_runs
