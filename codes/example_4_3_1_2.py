import tensorflow as tf
import numpy as np
import time
from functions import deep_optimal_stopping
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

T, N, K = 1, 50, 40
r, chi, beta = 0.06, 40, 0.4
lr_values = [0.05, 0.005, 0.0005]
mc_runs = 500


def g(s, x):
    return tf.exp(-r * s) \
        * tf.maximum(K - tf.exp((r - .5 * beta ** 2) * s + beta / np.sqrt(d) * tf.reduce_sum(x, axis=1, keepdims=True)) * chi, 0)


_file = open('example_4_3_1_2.csv', 'w')
_file.write('dim, run, mean, time\n')
for d in [1, 5, 10, 50, 100, 500, 1000]:
    if d <= 50:
        train_steps = 1500
        batch_size = 8192
    elif d <= 100:
        train_steps = 1800
        batch_size = 4096
    else:
        train_steps = 3000
        batch_size = 2048
    lr_boundaries = [int(train_steps / 3), int(2 * train_steps / 3)]
    for run in range(5):
        tf.compat.v1.reset_default_graph()
        t0 = time.time()
        neurons = [d + 50, d + 50, 1]
        W = tf.cumsum(tf.compat.v1.random_normal(
            shape=[batch_size, d, N],
            stddev=np.sqrt(T / N)), axis=2
        )
        t = tf.constant(np.linspace(start=T / N, stop=T, num=N,
                                    endpoint=True, dtype=np.float32))
        X = W
        px_mean = deep_optimal_stopping(
            X, t, N, g, neurons, batch_size,
            train_steps, mc_runs,
            lr_boundaries, lr_values, epsilon=0.001
        )
        t1 = time.time()
        _file.write('%i, %i, %f, %f\n' % (d, run + 1, px_mean, t1 - t0))
_file.close()
