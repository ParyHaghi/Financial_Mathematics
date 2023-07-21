import tensorflow as tf
import numpy as np
import time
from functions import deep_optimal_stopping
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

T, N, K = 3, 9, 100
r, delta, beta = 0.05, 0.1, 0.2
batch_size = 8192
lr_values = [0.05, 0.005, 0.0005]
mc_runs = 500


def g(s, x):
    return tf.exp(-r * s) \
        * tf.maximum(tf.reduce_max(x, axis=1, keepdims=True) - K, 0)


_file = open('example_4_4_1_1.csv', 'w')
_file.write('dim, run, mean, time\n')
for d in [2, 3, 5, 10, 20, 30, 50, 100, 200, 500]:
    for s_0 in [90, 100, 110]:
        for run in range(10):
            tf.compat.v1.reset_default_graph()
            t0 = time.time()
            neurons = [d + 50, d + 50, 1]
            train_steps = 3000 + d
            lr_boundaries = [int(500 + d / 5), int(1500 + 3 * d / 5)]
            W = tf.cumsum(tf.compat.v1.random_normal(
                shape=[batch_size, d, N],
                stddev=np.sqrt(T / N)), axis=2
            )
            t = tf.constant(np.linspace(start=T / N, stop=T, num=N,
                                        endpoint=True, dtype=np.float32))
            X = tf.exp((r - delta - beta ** 2 / 2) * t + beta * W) * s_0
            px_mean = deep_optimal_stopping(
                X, t, N, g, neurons, batch_size,
                train_steps, mc_runs,
                lr_boundaries, lr_values, epsilon=0.1
            )
            t1 = time.time()
            _file.write('%i, %i, %f, %f\n' % (d, run + 1, px_mean, t1 - t0))
_file.close()


