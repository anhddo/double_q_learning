# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model

class OptimisticCNN(Model):
    def __init__(self, n_action, beta):
        super(OptimisticCNN, self).__init__()
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
        self.dense1 = Dense(512, 
                activation='relu',
                kernel_initializer='he_normal')
        self.dense2 = Dense(n_action,
                kernel_initializer='he_normal')

        self.M = tf.Variable(tf.eye(512, batch_shape=[n_action]) * 10, trainable=False)
        self.beta = beta

    @tf.function
    def forward(self, x):
        x = tf.cast(x, tf.float32) / 255.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        embeded_vector = self.dense1(x)
        Q = self.dense2(embeded_vector) 
        bonus = self.bonus(embeded_vector)
        bonus = tf.stop_gradient(bonus)
        return embeded_vector, Q + bonus

    @tf.function
    def call(self, x):
        _, Q = self.forward(x)
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
        M_a.assign(M_a - self._update_term(M_a, s))
