# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model
import tensorflow as tf

class CNN(Model):
    def __init__(self, n_action):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(filters=32, 
                kernel_size=8,
                strides=4,
                activation='relu',
                kernel_initializer='he_normal')

        self.conv2 = Conv2D(filters=64, 
                kernel_size=4,
                strides=2,
                activation='relu',
                kernel_initializer='he_normal')

        self.conv3 = Conv2D(filters=64, 
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='he_normal')

        self.flatten = Flatten()
        self.hidden_size = 512
        self.dense1 = Dense(self.hidden_size, 
                activation='relu',
                kernel_initializer='he_normal')
        self.dense2 = Dense(n_action,
                kernel_initializer='he_normal')

    @tf.function
    def call(self, x):
        x = self.forward_dense1(x)
        return self.dense2(x)

    @tf.function
    def forward_dense1(self, x):
        x = tf.cast(x, tf.float32) / 255.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x 

    @tf.function
    def take_action(self, state):
        Q = self.call(state)
        action = tf.argmax(Q, axis=1, output_type=tf.dtypes.int32)
        action = tf.squeeze(action)
        return action 

class OptimisticCNN(CNN):
    def __init__(self, n_action, beta, args):
        super(OptimisticCNN, self).__init__(n_action)
        self.M = tf.Variable(tf.eye(self.hidden_size, batch_shape=[n_action]) * 10, trainable=False)
        self.M0 = tf.Variable(tf.eye(self.hidden_size, batch_shape=[n_action]) * 10, trainable=False)
        self.identity_matrix = tf.Variable(tf.eye(self.hidden_size) * 10, trainable=False)
        self.beta = beta
        self.index = tf.Variable(tf.zeros((n_action), dtype=tf.dtypes.int32), 
                trainable=False)
        self.buffer = args.buffer


    @tf.function
    def _call(self, x):
        embeded_vector = self.forward_dense1(x)
        Q = self.dense2(embeded_vector) 
        bonus = self.bonus(embeded_vector)
        bonus = tf.stop_gradient(bonus)
        return Q + bonus, embeded_vector

    @tf.function
    def call(self, x):
        Q, _ = self._call(x)
        return Q 

    @tf.function
    def bonus(self, embeded_vector):
        MX = tf.matmul(embeded_vector, self.M)
        bonus = tf.reduce_sum(tf.multiply(embeded_vector, MX), axis=2)
        bonus = tf.sqrt(bonus)
        bonus = self.beta * tf.transpose(bonus)
        return bonus

    @tf.function
    def _update_term(self, A, s):
        s = tf.reshape(s, shape=[-1, 1])
        v = tf.transpose(s) @ A
        return (A @ s @ v) / (1. + v @ s)

    @tf.function
    def update_inverse_covariance(self, a, s):
        M_a = self.M[a]
        delta_matrix = self._update_term(M_a, s)
        M_a.assign(M_a - delta_matrix)
        self.index[a].assign((self.index[a] + 1) %  self.buffer)
        if self.index[a] ==  0:
            self.M[a].assign(tf.identity(self.M0[a]))
            self.M0[a].assign(self.identity_matrix)
        M_a = self.M0[a]
        M_a.assign(M_a - delta_matrix)

    @tf.function
    def _take_action(self, state):
        Q, embeded_vector = self._call(state)
        action = tf.argmax(Q, axis=1, output_type=tf.dtypes.int32)
        action = tf.squeeze(action)
        return action, embeded_vector

    @tf.function
    def take_action(self, state):
        action, _ = self._take_action(state)
        return action

    @tf.function
    def take_action_train(self, state):
        action, embeded_vector = self._take_action(state)
        self.update_inverse_covariance(action, embeded_vector)
        return action
