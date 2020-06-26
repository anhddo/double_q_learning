import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, losses, optimizers, Sequential
from ..algo import EpsilonGreedy
from tqdm import trange
import pandas as pd
from datetime import datetime
from ..util import allow_gpu_growth
from collections import namedtuple, deque
import numpy.random as npr
import gym
import argparse
import sys
import os
import json
import glob
import time

allow_gpu_growth()

Sample = namedtuple('Sample', ('state', 'action', 'reward', 'next_state', 'done'))
#tf.config.experimental_run_functions_eagerly(True)

#tf.keras.backend.set_floatx('float64')
#TF_TYPE = tf.float64

#tf.keras.backend.set_floatx('float32')
TF_TYPE = tf.float32

class ReplayBuffer(object):
    def __init__(self, N):
        self.buffer = deque(maxlen=N)

    def add(self, sample):
        self.buffer.append(sample)

    #@tf.function
    def get_batch(self, batch_size):
        select_index = npr.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in select_index]
        return batch

class CNN(Model):
    def __init__(self, input_dim, out_dim):
        super(MLP, self).__init__()
        self.fn1 = Dense(32, activation='relu')
        self.fn2 = Dense(32, activation='relu')
        self.head = Dense(out_dim)

    @tf.function
    def call(self, x):
        x = self.fn1(x)
        x = self.fn2(x)
        return self.head(x)

class DDQN(object):
    def __init__(self, **kargs):
        obs_dim, action_dim = kargs['obs_dim'], kargs['action_dim']
        self.discount = kargs['discount']
        self.batch_size = kargs['batch_size']
        self.policy_net = MLP(obs_dim, action_dim)
        self.target_net = MLP(obs_dim, action_dim)
        self.policy_net.trainable = True
        self.target_net.trainable = False
        #self.optimizer = optimizers.Adam()
        self.optimizer = optimizers.RMSprop(learning_rate=0.000625)
        #self.loss_func = losses.Huber(tf.constant(1.0, dtype=TF_TYPE))
        self.loss_func = losses.Huber()
        #self.loss_func = losses.MeanSquaredError()
        self.policy_net(tf.random.uniform(shape=[1, obs_dim], dtype=TF_TYPE))
        self.target_net(tf.random.uniform(shape=[1, obs_dim], dtype=TF_TYPE))
        self.update()

    def update(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    @tf.function
    def _take_action(self, state):
        Q = self.policy_net(state)
        A = tf.argmax(Q, axis=1, output_type=tf.int64)
        return A

    @tf.function
    #def train(self, batch):
    def train(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            Q = self.policy_net(state)
            Q = tf.gather(Q, action, batch_dims=1)

            #action_next = self._take_action(next_state)
            ##
            Q1 = self.target_net(state)
            action_next = tf.argmax(Q1, axis=1)#, output_type=tf.int64)
            ##

            Q_target = self.target_net(next_state)
            V_target = tf.gather(Q_target, tf.reshape(action_next, shape=(-1, 1)), batch_dims=1)
            Q_target = reward + self.discount * tf.multiply(V_target, (1. - done))
            loss = self.loss_func(Q, Q_target)
        grad = tape.gradient(loss, self.policy_net.trainable_variables)
        grad = [tf.clip_by_value(e, -1., 1.) for e in grad]
        self.optimizer.apply_gradients(zip(grad, self.policy_net.trainable_variables))
        loss_info = {'loss': loss, 'Q': tf.reduce_max(Q), 'Q_target': tf.reduce_max(Q_target)}
        return loss_info


def train(setting):
    env = gym.make(setting['env'])

    A = env.action_space.n

    writer = tf.summary.create_file_writer('{}/logs/{}-{}'\
            .format(os.path.expanduser('~'),
                setting['env'],
                str(datetime.now())
            )
        )

    writer.set_as_default()

    state = env.reset()
    state = tf.constant(state, TF_TYPE)


    replay_buffer = ReplayBuffer(setting['buffer'])
    obs_dim = len(state)
    agent = DDQN(
            obs_dim=obs_dim, 
            action_dim=A,
            discount=setting['discount'],
            batch_size=setting['batch'], 
            replay_buffer=replay_buffer)

    agent.take_action = EpsilonGreedy(
            setting['max_epsilon'],
            setting['min_epsilon'],
            setting['step'], 
            setting['fraction'], 
            lambda :[env.action_space.sample()],
            lambda s: agent._take_action(s)).action

    ep_reward = 0
    tracking = []

    for step in trange(setting['step']):
        action, action_info = agent.take_action(tf.reshape(state, shape=[-1, obs_dim]))
        next_state, reward, done, _ = env.step(tf.squeeze(action).numpy())
        ep_reward += reward

        next_state, reward, terminal = tf.constant(next_state, dtype=TF_TYPE),\
                tf.constant([reward], dtype=TF_TYPE),\
                tf.constant([done], dtype=TF_TYPE) 
        sample = (state, action, reward, next_state, terminal)
        replay_buffer.add(sample)

        state = next_state
        if step > setting['start']:

            if step % setting['train_step'] == 0:
                batch = replay_buffer.get_batch(setting['batch'])
                state_, action_, reward_, next_state_, done_ = (tf.stack(e) for e in zip(*batch))
                train_info = agent.train(state_, action_, reward_, next_state_, done_ )

            if terminal:
                tf.summary.scalar('metrics/epsilon', data=action_info['epsilon'], step=step)
                if train_info:
                    tf.summary.scalar('metrics/loss', data=train_info['loss'], step=step)
                    tf.summary.scalar('metrics/Q', data=train_info['Q'], step=step)
                tf.summary.scalar('metrics/reward', data=ep_reward, step=step)

            if step % setting['update'] == 0:
                agent.update()

        if terminal:
            tracking.append([ep_reward, step])
            state = env.reset()
            state = tf.constant(state, TF_TYPE)
            ep_reward = 0

    files = glob.glob(os.path.join(setting['save_dir'],"*.csv"))
    csv_index = 1 + max([int(os.path.basename(file).split('.')[0]) for file in files] + [0])
    csv_path = os.path.join(setting['save_dir'], '{}.csv'.format(csv_index))
    reward, timestep = zip(*tracking)
    df = pd.DataFrame({'t': timestep, 'reward': reward})
    df.to_csv(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--fourier-order", type=int, default=1)
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--buffer", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save-dir")
    parser.add_argument("--n-run", type=int, default=10)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--update", type=int, default=100)
    parser.add_argument("--max-epsilon", type=float, default=0.8)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--train-step", type=int, default=5)
    parser.add_argument("--start", type=int, default=1000)

    args = parser.parse_args()
    setting = vars(args)

    os.makedirs(setting['save_dir'], exist_ok=True)
    with open(os.path.join(setting['save_dir'], 'setting.json'), 'w') as f:
        f.write(json.dumps(setting))

    for _ in range(setting['n_run']):
        train(setting)
