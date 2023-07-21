import tensorflow as tf
import numpy as np
import time
from functions import deep_optimal_stopping
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

tau, N = 30 / 365, 10
r, beta = 0.05 * tau, 0.4 * np.sqrt(tau)
batch_size = 8192
train_steps = 1600
lr_boundaries = [400, 800]
lr_values = [0.05, 0.005, 0.0005]
mc_runs = 500


def g(s, x, k):
    return tf.exp(-r * s) \
        * tf.maximum(tf.reduce_max(x, axis=1, keepdims=True) - k, 0)


_file = open('example_4_4_1_3.csv', 'w')
_file.write('dim, T, K, run, mean, time\n')
for d in [10, 400]:
    for T in [1, 4, 7] if d == 10 else [12]:
        for K in [35, 40, 45]:
            for run in range(10):
                tf.compat.v1.reset_default_graph()
                t0 = time.time()
                Q = np.ones([d, d], dtype=np.float32) * 0.5
                np.fill_diagonal(Q, 1)
                L = tf.constant(np.linalg.cholesky(Q).transpose())
                neurons = [d + 50, d + 50, 1]
                W = tf.matmul(tf.compat.v1.random_normal(
                    shape=[batch_size, N, d],
                    stddev=np.sqrt(T / N)), L
                )
                W = tf.cumsum(tf.transpose(tf.reshape(
                    W, [batch_size, N, d]),
                    [0, 2, 1]), axis=2
                )
                t = tf.constant(np.linspace(
                    start=T / N, stop=T, num=N,
                    endpoint=True, dtype=np.float32
                ))
                X = tf.exp((r - beta ** 2 / 2) * t + beta * W) * 40
                px_mean = deep_optimal_stopping(
                    X, t, N, lambda s, x: g(s, x, K),
                    neurons, batch_size, train_steps, mc_runs,
                    lr_boundaries, lr_values, epsilon=1e-3
                )
                t1 = time.time()
                _file.write('%i, %f, %f, %i, %f, %f\n'
                            % (d, T, K, run, px_mean, t1 - t0))
_file.close()
