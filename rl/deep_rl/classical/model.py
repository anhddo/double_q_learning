from tensorflow.keras import Model, losses, optimizers, Sequential
import tensorflow as tf, numpy as np
from tensorflow.keras.layers import Dense
from rl.algo import EpsilonGreedy
from rl.util import objectview
import numpy.random as npr

TF_TYPE = tf.float32


class MLP(Model):
    def __init__(self, input_dim, out_dim):
        super(MLP, self).__init__()
        self.hidden_size = 32
        self.fn1 = Dense(self.hidden_size, activation='relu')
        self.fn2 = Dense(self.hidden_size, activation='relu')
        self.head = Dense(out_dim)

    @tf.function
    def forward(self, x):
        x = self.fn1(x)
        embeded_vector = self.fn2(x)
        Q = self.head(embeded_vector) 
        return Q, embeded_vector

    @tf.function
    def call(self, x):
        Q, _ = self.forward(x)
        return Q

    @tf.function
    def take_action(self, state):
        Q = self.call(state)
        action = tf.argmax(Q, axis=1, output_type=tf.dtypes.int32)
        action = tf.squeeze(action)
        return action 

class OptimisticMLP(MLP):
    """
    out_dim: number of action
    """
    def __init__(self, input_dim, out_dim, beta):
        super(OptimisticMLP, self).__init__(input_dim, out_dim)
        self.M = tf.Variable(tf.eye(self.hidden_size, batch_shape=[out_dim]) * 10)
        self.beta = beta


    @tf.function
    def bonus(self, state):
        MX = tf.matmul(state, self.M)
        bonus = tf.reduce_sum(tf.multiply(state, MX), axis=2)
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

    @tf.function
    def _take_action(self, state):
        Q, embeded_vector = self.forward(state)
        bonus = self.bonus(embeded_vector)
        Q = Q + self.beta * bonus
        action = tf.argmax(Q, output_type=tf.dtypes.int32)
        tf.print(action)
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


class DDQN(object):
    def __init__(self, args):
        self.debug = args.debug
        obs_dim, action_dim = args.obs_dim, args.action_dim
        self.action_dim = action_dim
        self.discount = args.discount
        self.batch_size = args.batch

        self.use_huber = args.huber
        self.use_mse = args.mse


        self.use_rms = args.rms
        self.use_adam = args.adam

        self.tau = args.tau

        self.train_net = MLP(obs_dim, action_dim)
        self.fixed_net = MLP(obs_dim, action_dim)
        self.train_net.trainable = True
        self.fixed_net.trainable = False



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

        self.train_net(tf.random.uniform(shape=[1, obs_dim], dtype=TF_TYPE))
        self.fixed_net(tf.random.uniform(shape=[1, obs_dim], dtype=TF_TYPE))
        self.hard_update()
        self.e_greedy_train = EpsilonGreedy(args)
        self.eval_epsilon = args.eval_epsilon
        self.setup_network(args)

    def setup_network(self, args):
        self.policy_net = self.train_net
        if args.fixed_policy:
            self.policy_net = self.fixed_net

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

    def take_action_train(self, state, step):
        epsilon = self.e_greedy_train.get_epsilon(step)
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


    def train(self, batch):
        state, action, reward, next_state, done = zip(*batch)
        state = np.stack(state)
        next_state = np.stack(next_state)
        action, done, reward = np.array(action, dtype=np.int),\
                np.array(done, dtype=np.float32),\
                np.array(reward, dtype=np.float32)
        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)
        action, reward, done = action.reshape(-1, 1), reward.reshape(-1, 1), done.reshape(-1, 1)
        if self.debug:
            assert action.shape == (self.batch_size, 1)
            assert done.shape == (self.batch_size, 1)
            assert reward.shape == (self.batch_size, 1)


        
        return self.train_(state, action, reward, next_state, done)

    @tf.function
    def train_(self, state, action, reward, next_state, done):
        """
        state, next_state: 
        action, reward, done: 
        """
        with tf.GradientTape() as tape:
            Q = self.train_net(state)
            action_onehot = tf.one_hot(tf.squeeze(action), self.action_dim)
            Q = Q * action_onehot

            Q_next = self.next_policy_net(next_state)
            next_action = tf.argmax(Q_next, axis=1)
            next_action_onehot = tf.one_hot(next_action, self.action_dim)

            Q_next = Q_next * next_action_onehot
            V_next = tf.reduce_sum(Q_next, 1, keepdims=True)

            Q_next = self.Q_next_net(next_state)
            next_action_onehot = tf.one_hot(next_action, self.action_dim)
            Q_next = Q_next * next_action_onehot
            V_next = tf.reduce_sum(Q_next, 1, keepdims=True)
            Q_target = reward + self.discount * tf.stop_gradient(V_next) * (1. - done)
            loss = self.loss_func(Q, Q_target)
            if self.debug:
                pass
        grad = tape.gradient(loss, self.train_net.trainable_variables)
        #grad = [tf.clip_by_value(e, -1., 1.) for e in grad]
        self.optimizer.apply_gradients(zip(grad, self.train_net.trainable_variables))
        return {
                'loss': loss,
                'Q': tf.reduce_max(Q),
                'Q_target': tf.reduce_max(Q_target)
                }
class OptimisticDDQN(DDQN):
    def __init__(self, args):
        super(OptimisticDDQN, self).__init__(args)
        self.train_net = OptimisticMLP(args.obs_dim, args.action_dim, args.beta)
        self.fixed_net = OptimisticMLP(args.obs_dim, args.action_dim, args.beta)
        self.beta = args.beta
        self.setup_network(args)

    def take_action_train(self, state, step):
        return self.policy_net.take_action_train(state).numpy()

    def take_action_eval(self, state):
        return self.policy_net.take_action(state).numpy()


