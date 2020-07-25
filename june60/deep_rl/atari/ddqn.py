# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import tensorflow as tf
from tensorflow.keras import Model, losses, optimizers, Sequential
from .model import CNN

class DDQN(object):
    """
    Double Q learning
    """
    def __init__(self, args):
        action_dim = args.action_dim
        self.discount = args.discount
        self.batch_size = args.batch

        self.use_huber = args.huber
        self.use_mse = args.mse


        self.use_rms = args.rms
        self.use_adam = args.adam

        self.tau = args.tau

        self.train_net = CNN(action_dim)
        self.fixed_net = CNN(action_dim)
        self.train_net.trainable = True
        self.fixed_net.trainable = False

        if args.dqn:
            self.next_policy_net = self.fixed_net
            self.Q_next_net = self.fixed_net
        elif args.ddqn:
            if args.pol:
                self.next_policy_net = self.fixed_net
                self.Q_next_net = self.train_net
            elif args.vi:
                self.next_policy_net = self.train_net
                self.Q_next_net = self.fixed_net


        if args.rms:
            self.optimizer = optimizers.RMSprop(learning_rate=args.lr, rho=0.95, epsilon=0.01)
        elif args.adam:
            self.optimizer = optimizers.Adam(learning_rate=args.lr)
        elif args.sgd:
            self.optimizer = optimizers.SGD(learning_rate=args.lr, momentum=0.9)

        if args.huber:
            self.loss_func = losses.Huber()
        elif args.mse:
            self.loss_func = losses.MeanSquaredError()

        self.train_net(tf.random.uniform(shape=[1, 84, 84, 4]))
        self.fixed_net(tf.random.uniform(shape=[1, 84, 84, 4]))
        self.hard_update()

        self.policy_net = self.train_net
        if args.fixed_policy:
            self.policy_net = self.fixed_net


    def hard_update(self):
        self.fixed_net.set_weights(self.train_net.get_weights())

    def soft_update(self):
        weights = [fixed_weight * (1. - self.tau) + train_weight * self.tau \
                for train_weight, fixed_weight \
                in zip(
                 self.train_net.get_weights(),
                 self.fixed_net.get_weights())]
        self.fixed_net.set_weights(weights)

    def save_model(self, path):
        self.train_net.save_weights(path)

    def load_model(self, path):
        if path is not None:
            self.train_net.load_weights(path)
        else:
            print('No model path')

    @tf.function
    def _take_action(self, state):
        Q = self.policy_net(state)
        A = tf.argmax(Q, axis=1)
        return A

    def train(self, batch):
        state, action, reward, next_state, done = batch
        return self.train_(state, action, reward, next_state, done)

    @tf.function
    def train_(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            Q = self.train_net(state)
            Q = tf.gather(Q, action, batch_dims=1)

            Q_policy_next = self.next_policy_net(next_state)
            next_action = tf.argmax(Q_policy_next, axis=1)
            next_action = tf.reshape(next_action, shape=(-1, 1))

            Q_next = self.Q_next_net(next_state)
            V_next = tf.gather(Q_next, next_action, batch_dims=1)
            Q_target = reward + self.discount * tf.multiply(tf.stop_gradient(V_next), (1. - done))
            ##-----------------------CHECK TENSOR SHAPE-----------------------------##
            #tf.debugging.assert_equal(state.shape, (self.batch_size, 84, 84, 4))
            #tf.debugging.assert_equal(action.shape, (self.batch_size, 1))
            #tf.debugging.assert_equal(next_state.shape, (self.batch_size, 84, 84, 4))
            #tf.debugging.assert_equal(reward.shape, (self.batch_size, 1))
            #tf.debugging.assert_equal(done.shape, (self.batch_size, 1))
            #tf.debugging.assert_equal(next_action.shape, (self.batch_size, 1))
            #tf.debugging.assert_equal(V_next.shape, (self.batch_size, 1))
            ##______________________________________________________________________##
            #loss = tf.clip_by_value(self.loss_func(Q, Q_target), -1, 1)
            loss = self.loss_func(Q, Q_target)
        grad = tape.gradient(loss, self.train_net.trainable_variables)
        grad = [tf.clip_by_value(e, -1., 1.) for e in grad]
        self.optimizer.apply_gradients(zip(grad, self.train_net.trainable_variables))
        return {
                'loss': loss,
                'Q': tf.reduce_max(Q),
                'Q_target': tf.reduce_max(Q_target)
                }
