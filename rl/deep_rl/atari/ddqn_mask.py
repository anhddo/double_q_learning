# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import tensorflow as tf
from tensorflow.keras import Model, losses, optimizers, Sequential
from .model import CNN, OptimisticCNN
from rl.algo import EpsilonGreedy
import numpy.random as npr

class DDQN_Base(object):
    """
    Double Q learning
    """
    def __init__(self, train_net, fixed_net, args):
        self.debug = args.debug
        self.clip_grad = args.clip_grad
        self.action_dim = args.action_dim
        self.discount = args.discount
        self.batch_size = args.batch

        self.use_huber = args.huber
        self.use_mse = args.mse


        self.use_rms = args.rms
        self.use_adam = args.adam

        self.tau = args.tau

        self.train_net = train_net
        self.fixed_net = fixed_net 



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
            self.optimizer = optimizers.RMSprop(
                    learning_rate=args.lr,
                    rho=0.95,
                    momentum=0.95,
                    epsilon=0.01)
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

        self.train_info = None


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
    def _take_action0(self, state):
        embeded_vector, Q = self.policy_net.forward(state)
        A = tf.argmax(Q, axis=1)
        self.policy_net.update_inverse_covariance(tf.squeeze(A), embeded_vector)
        return A

    @tf.function
    def _take_action(self, state):
        Q = self.policy_net(state)
        A = tf.argmax(Q, axis=1)
        return A

    def train(self, batch):
        state, action, reward, next_state, done = batch
        self.train_info = self.train_(state, action, reward, next_state, done)
        return self.train_info

    @tf.function
    def train_(self, state, action, reward, next_state, done):
        """
        state, next_state: [batch_size, 84, 84, 4]
        action, reward, done, V_next: [batch_size, 1]
        """
        with tf.GradientTape() as tape:
            Q = self.train_net(state)
            action_onehot = tf.one_hot(tf.squeeze(action), self.action_dim)
            Q = Q * action_onehot
            Q = tf.reduce_sum(Q, 1, keepdims=True)

            Q_policy_next = self.next_policy_net(next_state)
            next_action = tf.argmax(Q_policy_next, axis=1)

            Q_next = self.Q_next_net(next_state)
            next_action_onehot = tf.one_hot(next_action, self.action_dim)
            Q_next = Q_next * next_action_onehot
            V_next = tf.reduce_sum(Q_next, 1, keepdims=True)
            Q_target = reward + self.discount * tf.multiply(tf.stop_gradient(V_next), (1. - done))
            ##-----------------------CHECK TENSOR SHAPE-----------------------------##
            if self.debug:
                tf.debugging.assert_equal(state.shape, (self.batch_size, 84, 84, 4))
                tf.debugging.assert_equal(action.shape, (self.batch_size, 1))
                tf.debugging.assert_equal(next_state.shape, (self.batch_size, 84, 84, 4))
                tf.debugging.assert_equal(reward.shape, (self.batch_size, 1))
                tf.debugging.assert_equal(done.shape, (self.batch_size, 1))
                tf.debugging.assert_equal(next_action.shape, (self.batch_size))
                tf.debugging.assert_equal(V_next.shape, (self.batch_size, 1))
                tf.debugging.assert_equal(Q.shape, (self.batch_size, 1))
                tf.debugging.assert_equal(Q_target.shape, (self.batch_size, 1))
            ##______________________________________________________________________##
            loss = self.loss_func(Q, Q_target)
        grad = tape.gradient(loss, self.train_net.trainable_variables)
        if self.clip_grad:
            grad = [tf.clip_by_value(e, -1., 1.) for e in grad]

        grad = [egrad if egrad is not None else tf.zeros_like(trainable_var) for egrad, trainable_var in zip(grad, self.train_net.trainable_variables) ]
        self.optimizer.apply_gradients(zip(grad, self.train_net.trainable_variables))
        return {
                'loss': loss,
                'Q': tf.reduce_max(Q),
                'Q_target': tf.reduce_max(Q_target)
                }

    def log(self, logs, step):
        if self.train_info:
            logs.loss.append((step, round(float(self.train_info['loss'].numpy()), 3)))
            logs.Q.append((step, round(float(self.train_info['Q'].numpy()), 3)))

            tf.summary.scalar("train/Q", data=self.train_info['Q'], step=step)
            tf.summary.scalar("train/loss", data=self.train_info['loss'], step=step)


class DDQN_Epsilon_Greedy(DDQN_Base):
    def __init__(self, args):
        train_net = CNN(args.action_dim)
        fixed_net = CNN(args.action_dim)
        super(DDQN_Epsilon_Greedy, self).__init__(train_net, fixed_net, args)
        self.e_greedy_train = EpsilonGreedy(args)
        self.eval_epsilon = args.eval_epsilon


    def take_action_train(self, state, step):
        epsilon = self.e_greedy_train.get_epsilon(step)
        self.train_epsilon = epsilon
        if epsilon > npr.uniform():
            return npr.randint(0, self.action_dim)
        else:
            return self.policy_net.take_action(state).numpy()

    def take_action_eval(self, state):
        if self.eval_epsilon > npr.uniform():
            return npr.randint(0, self.action_dim)
        else:
            action = self.policy_net.take_action(state).numpy()
            return action

    def train_info_str(self):
        return "Epsilon: {}".format(self.train_epsilon) 


class OptimisticDDQN(DDQN_Base):
    def __init__(self, args):
        train_net = OptimisticCNN(args.action_dim, args.beta, args)
        fixed_net = OptimisticCNN(args.action_dim, args.beta, args)
        self.beta = args.beta
        super(OptimisticDDQN, self).__init__(train_net, fixed_net, args)

    def take_action_train(self, state, step):
        return self.policy_net.take_action_train(state).numpy()

    def take_action_eval(self, state):
        return self.policy_net.take_action(state).numpy()

    def train(self, batch):
        #state, action, reward, next_state, done = self.train_preprocess(batch)
        state, action, reward, next_state, done = batch
        self.train_info = self.train_(state, action, reward, next_state, done)
        _, z = self.policy_net._call(state) 
        self.train_info['bonus'] = tf.reduce_mean(self.policy_net.bonus(z))
        self.train_info['z'] = tf.reduce_max(tf.abs(z))
        self.train_info['cov'] = tf.reduce_max(tf.abs(self.policy_net.M))


    def log(self, logs, step):
        super(OptimisticDDQN, self).log(logs, step)
        tf.summary.scalar("train/bonus", data=self.train_info['bonus'], step=step)
        tf.summary.scalar("train/z", data=self.train_info['z'], step=step)
        tf.summary.scalar("train/cov", data=self.train_info['cov'], step=step)

    #@tf.function
    #def train_(self, state, action, reward, next_state, done):
    #    tracking = super(OptimisticDDQN, self).train_(, action, reward, next_state, done)


