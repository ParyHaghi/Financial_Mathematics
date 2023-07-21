import tensorflow as tf
import numpy as np
import time
from functions import deep_optimal_stopping
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

d, r = 100, 0.0004
sigma = 0.02
batch_size = 8192
neurons = [150, 150, 1]
train_steps = 1200
lr_boundaries = [400, 800]
lr_values = [0.05, 0.005, 0.0005]
mc_runs = 500


def g(s, x):
    return tf.exp(-r * s) * tf.expand_dims(x[:, 99, :], axis=1)


_file = open('example_4_4_4.csv', 'w')
_file.write('N, run, mean, time\n')
for N in [100, 150, 250, 1000, 2000]:
    for run in range(10):
        tf.compat.v1.reset_default_graph()
        t0 = time.time()
        T = np.float32(N)
        W = tf.compat.v1.random_normal(
            shape=[batch_size, N + 100],
            stddev=np.sqrt((N + 100) / N)
        )
        W = tf.cumsum(tf.concat([tf.zeros([batch_size, 1]),
                                 W], axis=1), axis=1)
        _W = tf.reshape(W[:, 1:], [batch_size, N + 100, 1, 1])
        patches = tf.compat.v1.extract_image_patches(
            _W, ksizes=[1, d, 1, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        W = tf.squeeze(patches) - tf.expand_dims(W[:, :N + 1], axis=2)
        W = tf.transpose(W, [0, 2, 1])
        n = tf.expand_dims(tf.constant(np.linspace(
            start=1, stop=d, num=d,
            endpoint=True,
            dtype=np.float32)), axis=1)
        X = tf.exp((r - sigma ** 2 / 2) * n + sigma * W)
        t = tf.constant(np.linspace(start=0, stop=T, num=N + 1,
                                    endpoint=True, dtype=np.float32))
        px_mean = deep_optimal_stopping(X, t, N + 1, g,
                                        neurons, batch_size,
                                        train_steps, mc_runs,
                                        lr_boundaries, lr_values)
        t1 = time.time()
        _file.write('%i, %i, %f, %f\n' % (N, run, px_mean, t1 - t0))
        _file.flush()
_file.close()
