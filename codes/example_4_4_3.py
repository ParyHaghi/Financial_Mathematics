import tensorflow as tf
import numpy as np
import time
from functions import deep_optimal_stopping
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

T, L = 1., 10
d, r = 5, 0.05
delta, K = 0.1, 100.
neurons = [50, 50, 1]
train_steps = 1200
lr_boundaries = [400, 800]
lr_values = [0.05, 0.005, 0.0005]


def g(s, x):
    return tf.exp(-r * s) * tf.maximum(
        K - tf.reduce_mean(tf.exp(x), axis=1, keepdims=True), 0.
    )


_file = open('example_4_4_3.csv', 'w')
_file.write('L, N, delta, type, run, mean, time\n')
for N in [5, 10, 50, 100]:
    for delta in [0., 0.1]:
        for _type in ['A', 'E'] if N == 10 else ['A']:
            for run in range(10):
                tf.compat.v1.reset_default_graph()
                t0 = time.time()
                if _type == 'A':
                    batch_size, mc_runs = 8192, 500
                else:
                    batch_size, mc_runs = 100000, 10000

                W = tf.compat.v1.random_normal(shape=[batch_size, d, L],
                                               stddev=np.sqrt(T / L))
                X = []
                Y = tf.constant(np.log(100.), tf.float32)

                for l in range(L):
                    beta = 0.6 * np.exp(
                        -0.05 * np.sqrt(l * T / L)) \
                           * (1.2 - tf.exp(-0.1 * l * T / L - 0.001
                                           * (np.exp(r * l * T / L)
                                              * tf.exp(Y) - 100.) ** 2))
                    _b = (r - delta - beta ** 2 / 2.)
                    _w = beta * W[:, :, 1]
                    if N >= L:
                        X.append(Y)
                        for n in range(int(N / L - 1)):
                            _x = Y + (n + 1) * T / N * _b \
                                 + (n + 1.) * L / N * _w
                            X.append(_x)
                    elif l % (L / N) == 0:
                        X.append(Y)
                    Y += T / L * _b + _w

                if _type == 'A':
                    X.append(Y)
                    X = tf.stack(X[1:], axis=2)
                    t = tf.constant(np.linspace(
                        start=T / N, stop=T, num=N,
                        endpoint=True, dtype=np.float32
                    ))

                    px_mean = deep_optimal_stopping(X, t, N, g,
                                                    neurons,
                                                    batch_size,
                                                    train_steps,
                                                    mc_runs,
                                                    lr_boundaries,
                                                    lr_values)
                else:
                    t = tf.constant(T)
                    px = tf.reduce_mean(tf.maximum(g(t, Y), 0))

                    with tf.compat.v1.Session() as sess:
                        px_mean = 0.
                        for k in range(mc_runs):
                            px_mean += sess.run(px)
                        px_mean /= mc_runs

                t1 = time.time()
                _file.write('%i, %i, %f, %s, %i, %f, %f\n'
                            % (L, N, delta, _type, run,
                               px_mean, t1 - t0))
                _file.flush()
_file.close()
