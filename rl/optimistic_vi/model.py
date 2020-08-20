import tensorflow as tf
import numpy as np
import numpy.random as npr
import gym
from datetime import datetime
from rl.algo import EpsilonGreedy

class ValueIteration:
    """
    action_dim       is the number of action,
    buff_size       is the size of buffer, where we store all the past data.
    ftr_dim       is the dimension of feature

    Size of tensor:

        self.M      action_dim x ftr_dim x ftr_dim       Inverse covariance matrix
        self.X      action_dim x buff_size x ftr_dim       Current state data
        self.X1     action_dim x buff_size x ftr_dim       Next state data
        self.D      action_dim x buff_size           Terminal flag for an episode
        self.R      action_dim x buff_size           Reward for an episode
        self.w      action_dim x ftr_dim           Model parameter

    """
    def __init__(self, args):
        action_dim = args.action_dim
        ftr_dim = args.ftr_dim
        buff_size = args.buffer
        tf_type = tf.dtypes.double
        self.M = tf.Variable(tf.eye(ftr_dim, batch_shape=[action_dim], dtype=tf_type) * 10)
        self.X = tf.Variable(tf.zeros((action_dim, buff_size, ftr_dim), dtype=tf_type))
        self.X1 = tf.Variable(tf.zeros((action_dim, buff_size, ftr_dim), dtype=tf_type))
        self.R = tf.Variable(tf.zeros((action_dim, buff_size), dtype=tf_type))
        self.D = tf.Variable(tf.zeros((action_dim, buff_size), dtype=tf_type))
        self.w = tf.Variable(tf.zeros((action_dim, ftr_dim), dtype=tf_type))

        self.min_clip = tf.constant(args.min_clip, dtype=tf_type)
        self.max_clip = tf.constant(args.max_clip, dtype=tf_type)

        self.index = tf.Variable(tf.zeros((action_dim), dtype=tf.dtypes.int32))

        self.action_dim = action_dim
        self.buff_size = buff_size
        self.ftr_dim = ftr_dim
        #self.beta = tf.constant(beta, dtype=tf_type)

    @tf.function
    def observe(self, s, a, r, s1, done):
        index_ = self.index[a]
        self.X[a, index_, :].assign(s)
        self.X1[a, index_, :].assign(s1)
        self.R[a, index_].assign(r)
        self.D[a, index_].assign(done)
        self.index[a].assign((self.index[a] + 1) % self.buff_size)

    def take_action_train(self, state, step):
        action_index = self.take_action(state).numpy()
        self.update_inverse_covariance(action_index, state)
        return action_index

    @tf.function
    def take_action(self, state):
        V = tf.squeeze(tf.linalg.matvec(self.w, state))
        return tf.argmax(V, output_type=tf.dtypes.int32)


    @tf.function
    def _update_term(self, A, s):
        s = tf.reshape(s, shape=[-1, 1])
        v = tf.transpose(s) @ A
        return (A @ s @ v) / (1. + v @ s)

    @tf.function
    def update_inverse_covariance(self, a, s):
        index_ = self.index[a]
        x = self.X[a, index_, :]
        M_a = self.M[a]
        M_a.assign(M_a + self._update_term(M_a, x) - self._update_term(M_a, s))

    @tf.function
    def V1(self):
        V1 = tf.matmul(self.X1, self.w, transpose_b=True)
        V1 = tf.reduce_max(V1, axis=2)
        return V1

    @tf.function
    def update_w(self, y):
        y = tf.clip_by_value(y, clip_value_min=self.min_clip, clip_value_max=self.max_clip)
        X_T = tf.transpose(self.X, perm=[0, 2, 1])
        X_Ty = tf.linalg.matvec(X_T, y)
        self.w.assign(tf.linalg.matvec(self.M, X_Ty))

    @tf.function
    def train(self):
        V1 = self.V1()
        y = self.R + V1 * (1. - self.D)
        self.update_w(y)


class GreedyValueIteration(ValueIteration):
    def __init__(self, args):
        super(GreedyValueIteration, self).__init__(args)
        self.e_greedy_train = EpsilonGreedy(args)

    def take_action_train(self, state, step):
        epsilon = self.e_greedy_train.get_epsilon(step)
        if epsilon > npr.uniform():
            action_index = npr.randint(0, self.action_dim)
        else:
            action_index = self.take_action(state).numpy()
        self.update_inverse_covariance(action_index, state)
        return action_index



class OptimisticValueIteration(ValueIteration):
    def __init__(self, args):
        super(OptimisticValueIteration, self).__init__(args)
        self.beta = args.beta


    @tf.function
    def bonus(self, state):
        #s = tf.reshape(s, (-1, self.ftr_dim))
        if len(state.shape) == 1:
            bonus = tf.linalg.matvec(tf.linalg.matvec(self.M, state), state)
            return bonus
        MX = tf.matmul(state, self.M)
        bonus = tf.reduce_sum(tf.multiply(state, MX), axis=2)
        bonus = tf.math.sqrt(bonus)
        return bonus

    @tf.function
    def take_action(self, state):
        bonus = self.bonus(state)
        V = tf.squeeze(tf.linalg.matvec(self.w, state))
        V += self.beta * bonus
        action_index = tf.argmax(V, output_type=tf.dtypes.int32)
        return action_index

    @tf.function
    def train(self):
        V1 = self.V1()
        bonus = self.beta * self.bonus(self.X)
        y = self.R + (V1 + bonus) * (1. - self.D)
        self.update_w(y)
