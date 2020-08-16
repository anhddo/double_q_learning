# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model

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
        self.dense1 = Dense(512, 
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


