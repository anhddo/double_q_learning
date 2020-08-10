import tensorflow as tf
import numpy as np
import numpy.random as npr
import gym
from datetime import datetime

class FourierBasis(object):
    """
    We featurize the given observation using
    Value Function Approximation in Reinforcement Learning using the Fourier Basis, 
    George Konidaris, etal
    """
    def __init__(self, fourier_order, env):
        tf_type = tf.dtypes.double
        if env.spec._env_name == 'CartPole':
            min_data = tf.constant([-4.8, -4, -0.41, -4], dtype=tf_type)
            max_data = tf.constant([+4.8, +4, +0.41, +4], dtype=tf_type)
            min_clip, max_clip = [0, 500]
        elif env.spec._env_name == 'MountainCar':
            min_data = tf.constant([-1.2, -0.07], dtype=tf_type)
            max_data = tf.constant([0.6, 0.07], dtype=tf_type)
            min_clip, max_clip = [-200, 0]
        elif env.spec._env_name == 'Acrobot':
            min_data = tf.constant([-1., -1., -1., -1., -13.0, -22], dtype=tf_type)
            max_data = tf.constant([+1., +1., +1., +1., +13.0, +22], dtype=tf_type)
            min_clip, max_clip = [-500, 0]

        s = env.reset()
#         with tf.device("GPU:0"):
        d = (fourier_order + 1) ** len(s)
        fourier_kernel_mat = tf.reshape(
                tf.meshgrid(*([np.arange(fourier_order + 1)] * len(s))),
                shape=[-1, d]
            )
        fourier_kernel_mat = tf.transpose(fourier_kernel_mat)
        fourier_kernel_mat = tf.cast(fourier_kernel_mat, dtype=tf_type)
        self.range_data = min_data - max_data
        self.min_data = min_data
        self.max_data = max_data
        self.fourier_kernel_mat = fourier_kernel_mat
        env.close()

    @tf.function
    def transform(self, s):
        s = (s - self.min_data) / self.range_data
        return tf.math.cos(np.pi * tf.linalg.matvec(self.fourier_kernel_mat, s))

class OVI:
    """
    A       is the number of action,
    n       is the size of buffer, where we store all the past data.
    d       is the dimension of feature

    Size of tensor:

        self.M      A x d x d       Inverse covariance matrix
        self.X      A x n x d       Current state data
        self.X1     A x n x d       Next state data
        self.D      A x n           Terminal flag for an episode
        self.R      A x n           Reward for an episode
        self.w      A x d           Model parameter

    """
    def __init__(self, A, n, d, min_clip, max_clip, beta, env):
        tf_type = tf.dtypes.double
        self.M = tf.Variable(tf.eye(d, batch_shape=[A], dtype=tf_type) * 10)
        self.X = tf.Variable(tf.zeros((A, n, d), dtype=tf_type))
        self.X1 = tf.Variable(tf.zeros((A, n, d), dtype=tf_type))
        self.R = tf.Variable(tf.zeros((A, n), dtype=tf_type))
        self.D = tf.Variable(tf.zeros((A, n), dtype=tf_type))
        self.w = tf.Variable(tf.zeros((A, d), dtype=tf_type))

        self.min_clip = tf.constant(min_clip, dtype=tf_type)
        self.max_clip = tf.constant(max_clip, dtype=tf_type)

        #self.index = [0] * A #tf.Variable(tf.zeros((A), dtype=tf.dtypes.int32))
        self.index = tf.Variable(tf.zeros((A), dtype=tf.dtypes.int32))

        self.env = env

        self.A = A
        self.n = n
        self.beta = tf.constant(beta, dtype=tf_type)

    @tf.function
    def _update_term(self, A, s):
        s = tf.reshape(s, shape=[-1, 1])
        v = tf.transpose(s) @ A
        return (A @ s @ v) / (1. + v @ s)


    @tf.function
    def _predict(self):
        V1 = tf.matmul(self.X1, self.w, transpose_b=True)
        V1 = tf.reduce_max(V1, axis=2)
        V1 = tf.clip_by_value(V1, clip_value_min=self.min_clip, clip_value_max=self.max_clip)
        y = self.R + V1 * (1. - self.D)
        return y


    @tf.function
    def _take_action(self, s):
        bonus = tf.linalg.matvec(tf.linalg.matvec(self.M, s), s)
        V = tf.squeeze(tf.linalg.matvec(self.w, s)) + self.beta * bonus
        return tf.argmax(V, output_type=tf.dtypes.int32)

    @tf.function
    def observe(self, s, a, r, s1, done):
        index_ = self.index[a]
        self.X[a, index_, :].assign(s)
        self.X1[a, index_, :].assign(s1)
        self.R[a, index_].assign(r)
        self.D[a, index_].assign(done)
        self.index[a].assign((self.index[a] + 1) % self.n)

    @tf.function
    def update_inverse_covariance(self, a, s):
        index_ = self.index[a]
        x = self.X[a, index_, :]
        M_a = self.M[a]
        M_a.assign(M_a + self._update_term(M_a, x) - self._update_term(M_a, s))

    @tf.function
    def train(self):
        MX = tf.matmul(self.X, self.M)
        bonus = tf.reduce_sum(tf.multiply(self.X, MX), axis=2)
        bonus = tf.math.sqrt(bonus)
        y = self._predict() + self.beta * bonus
        X_T = tf.transpose(self.X, perm=[0, 2, 1])
        X_Ty = tf.linalg.matvec(X_T, y)
        self.w.assign(tf.linalg.matvec(self.M, X_Ty))

def record_video(agent, fourier_basis, env_name):
    env = gym.wrappers.Monitor(gym.make(env_name), './recording/' + env_name, force=True)
    s = env.reset()
    s = fourier_basis.transform(tf.constant(s, dtype=tf.dtypes.double))
    while True:
        s,_,done,_ = env.step(agent.take_action(s).numpy())
        s = fourier_basis.transform(tf.constant(s, dtype=tf.dtypes.double))
        if done:
            break
    env.close()
