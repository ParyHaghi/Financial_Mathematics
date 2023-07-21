import tensorflow as tf
import numpy as np
import time
from functions import deep_optimal_stopping
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

T, N, K = 3, 9, 100
r, delta, beta = 0.05, 0.1, 0.2
batch_size = 1024
lr_values = [0.01, 0.001, 0.0001]
mc_runs = 500
d = 5000


def g(s, x):
    return tf.exp(-r * s) \
        * tf.maximum(tf.reduce_max(x, axis=1, keepdims=True) - K, 0)


_file = open('example_4_4_1_2.csv', 'w')
_file.write('M, mean, time\n')

for m in range(8):
    tf.compat.v1.reset_default_graph()
    t0 = time.time()
    neurons = [d + 50, d + 50, 1]
    train_steps = 250 * m
    lr_boundaries = [2000, 4000]
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
        lr_boundaries, lr_values, epsilon=1e-8
    )
    t1 = time.time()
    _file.write('%i, %f, %f\n' % (train_steps, px_mean, t1 - t0))
_file.close()