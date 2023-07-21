import tensorflow as tf
import numpy as np
import time
from functions import deep_optimal_stopping
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

T, N = 1, 48
d, r = 5, 0.05
batch_size = 8192
neurons = [50, 50, 1]
train_steps = 750
lr_boundaries = [250, 500]
lr_values = [0.05, 0.005, 0.0005]
mc_runs = 500
Q = np.array([[0.3024, 0.1354, 0.0722, 0.1367, 0.1641],
              [0.1354, 0.2270, 0.0613, 0.1264, 0.1610],
              [0.0722, 0.0613, 0.0717, 0.0884, 0.0699],
              [0.1367, 0.1264, 0.0884, 0.2937, 0.1394],
              [0.1641, 0.1610, 0.0699, 0.1394, 0.2535]], dtype=np.float32)


def g(s, x):
    x = tf.reduce_mean(x, axis=1, keepdims=True)
    return tf.exp(-r * s) * (-tf.maximum(75 - x, 0)
                             + tf.maximum(90 - x, 0)
                             + tf.maximum(x - 110, 0)
                             - tf.maximum(x - 125, 0))


_file = open('example_4_4_2.csv', 'w')
_file.write('run, mean, time\n')
for run in range(10):
    tf.compat.v1.reset_default_graph()
    t0 = time.time()
    sigma = tf.expand_dims(tf.constant(np.linalg.norm(Q, axis=1)), axis=1)
    W = tf.matmul(tf.compat.v1.random_normal(shape=[batch_size * N, d],
                                             stddev=np.sqrt(T / N)), Q)
    W = tf.cumsum(tf.transpose(tf.reshape(W, [batch_size, N, d]),
                               [0, 2, 1]), axis=2)
    t = tf.constant(np.linspace(start=T / N, stop=T, num=N,
                                endpoint=True, dtype=np.float32))
    X = tf.exp((r - sigma ** 2 / 2) * t + W) * 100
    px_mean = deep_optimal_stopping(X, t, N, g,
                                    neurons, batch_size,
                                    train_steps, mc_runs,
                                    lr_boundaries, lr_values)
    t1 = time.time()
    _file.write('%i, %f, %f\n' % (run, px_mean, t1 - t0))
_file.close()
