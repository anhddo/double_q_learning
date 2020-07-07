# ======================== 
# Author: Anh Do
# ========================

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model, losses, optimizers, Sequential
from ..algo import EpsilonGreedy
from tqdm import trange, tqdm
from datetime import datetime
from ..util import allow_gpu_growth, Logs
from collections import namedtuple, deque
from os.path import join, isdir
import numpy.random as npr
import gym
import argparse
import sys
import os
import json
import time
from PIL import Image
import numpy as np

allow_gpu_growth()

Sample = namedtuple('Sample', ('state', 'action', 'reward', 'next_state', 'done'))
#tf.config.experimental_run_functions_eagerly(True)

#tf.keras.backend.set_floatx('float64')
#TF_TYPE = tf.float64

#tf.keras.backend.set_floatx('float32')
TF_TYPE = tf.float32

class RingBuffer(object):
    def __init__(self, N):
        self.buffer = deque(maxlen=N)

    def add(self, sample):
        self.buffer.append(sample)

    def get_batch(self, batch_size):
        select_index = npr.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in select_index]
        return batch

class RingBuffer1(object):
    def __init__(self, N):
        self.buffer = []
        self.N = N
        self.last_index = 0
        self.index = 0

    def add(self, sample):
        self.buffer.append(sample)
        self.last_index = min(self.last_index + 1, self.N)
        if self.last_index < self.N:
            self.buffer.append(sample)
        else:
            self.index = (self.index + 1) % self.N
            self.buffer[self.index] = sample

    def get_batch(self, batch_size):
        select_index = npr.choice(self.last_index, batch_size)
        batch = [self.buffer[i] for i in select_index]
        return batch

class CNN(Model):
    def __init__(self, n_action):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4,  activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2,  activation='relu')
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1,  activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(n_action)

    @tf.function
    def call(self, x):
        x = tf.cast(x, TF_TYPE) / 255.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class DDQN(object):
    def __init__(self, kargs):
        action_dim = kargs['action_dim']
        self.discount = kargs['discount']
        self.batch_size = kargs['batch']

        self.use_huber = kargs['huber']
        self.use_mse = kargs['mse']


        self.use_rms = kargs['rms']
        self.use_adam = kargs['adam']

        self.tau = kargs['tau']
        self.tboard = kargs['tboard']

        self.train_net = CNN(action_dim)
        self.fixed_net = CNN(action_dim)
        self.train_net.trainable = True
        self.fixed_net.trainable = False

        if kargs['dqn']:
            self.next_policy_net = self.fixed_net
            self.Q_next_net = self.fixed_net
        elif kargs['ddqn']:
            if kargs['pol']:
                self.next_policy_net = self.fixed_net
                self.Q_next_net = self.train_net
            elif kargs['vi']:
                self.next_policy_net = self.train_net
                self.Q_next_net = self.fixed_net


        if kargs['rms']:
            self.optimizer = optimizers.RMSprop(learning_rate=kargs['lr'], rho=0.95, epsilon=0.01)
        elif kargs['adam']:
            self.optimizer = optimizers.Adam(learning_rate=kargs['lr'])
        elif kargs['sgd']:
            self.optimizer = optimizers.SGD(learning_rate=kargs['lr'], momentum=0.9)

        #self.loss_func = losses.Huber(tf.constant(1.0, dtype=TF_TYPE))
        if kargs['huber']:
            self.loss_func = losses.Huber()
        elif kargs['mse']:
            self.loss_func = losses.MeanSquaredError()

        self.train_net(tf.random.uniform(shape=[1, 84, 84, 4], dtype=TF_TYPE))
        self.fixed_net(tf.random.uniform(shape=[1, 84, 84, 4], dtype=TF_TYPE))
        self.hard_update()

        self.policy_net = self.train_net
        if kargs['fixed_policy']:
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

    @tf.function
    def _take_action(self, state):
        Q = self.policy_net(state)
        A = tf.argmax(Q, axis=1)
        return A

    def train(self, batch):
        state, action, reward, next_state, done = zip(*batch)
        state = [np.stack(e, axis=2) for e in state]
        next_state = [np.stack(e, axis=2) for e in next_state]
        state, action, reward, next_state, done =\
                (np.stack(e) for e in (state, action, reward, next_state, done))
        reward = reward.reshape(-1, 1)
        done = done.reshape(-1, 1)
        return self.train_(state, action, reward, next_state, done)

    @tf.function
    def train_(self, state, action, reward, next_state, done):
        state, reward, next_state, done = [tf.cast(e, TF_TYPE) for e in \
            (state, reward, next_state, done)]
        with tf.GradientTape() as tape:
            Q = self.train_net(state)
            Q = tf.gather(Q, action, batch_dims=1)

            Q_policy_next = self.next_policy_net(next_state)
            next_action = tf.argmax(Q_policy_next, axis=1)
            next_action = tf.reshape(next_action, shape=(-1, 1))

            Q_next = self.Q_next_net(next_state)
            V_next = tf.gather(Q_next, next_action, batch_dims=1)
            Q_target = reward + self.discount * tf.multiply(V_next, (1. - done))
            ##-----------------------CHECK TENSOR SHAPE-----------------------------##
            tf.debugging.assert_equal(state.shape, (self.batch_size, 84, 84, 4))
            tf.debugging.assert_equal(action.shape, (self.batch_size, 1))
            tf.debugging.assert_equal(next_state.shape, (self.batch_size, 84, 84, 4))
            tf.debugging.assert_equal(reward.shape, (self.batch_size, 1))
            tf.debugging.assert_equal(done.shape, (self.batch_size, 1))
            tf.debugging.assert_equal(next_action.shape, (self.batch_size, 1))
            tf.debugging.assert_equal(V_next.shape, (self.batch_size, 1))
            ##______________________________________________________________________##

            loss = self.loss_func(Q, Q_target)
        grad = tape.gradient(loss, self.train_net.trainable_variables)
        #grad = [tf.clip_by_value(e, -1., 1.) for e in grad]
        self.optimizer.apply_gradients(zip(grad, self.train_net.trainable_variables))
        return {
                'loss': loss,
                'Q': tf.reduce_max(Q),
                'Q_target': tf.reduce_max(Q_target)
                }

#i=1
#class Preprocess(object):
#    def atari(self, img):
#        global i
#        im = Image.fromarray(img)\
#                .convert("L")
#        i+=1
#        im.save('image/{}.png'.format(i))
#        im = im\
#                .resize((84, 110), Image.NEAREST)\
#                .crop((0, 18, 84, 102))
#        im.save('image/im_crop{}.png'.format(i))
#        return np.asarray(im)
#
#class Preprocess(object):
#    def atari(self, img):
#        im = Image.fromarray(img)\
#                .convert("L")\
#                .resize((84, 110), Image.NEAREST)\
#                .crop((0, 18, 84, 102))
#        return np.asarray(im)
class Preprocess(object):
    def atari(self, img):
        im = Image.fromarray(img)\
                .convert("L")\
                .resize((84, 84), Image.NEAREST)
                #.crop((0, 18, 84, 102))
        return np.asarray(im)


def train(setting):
    if setting['tboard']:
        writer = tf.summary.create_file_writer('{}/logs/{}-{}'\
                .format(os.path.expanduser('~'),
                    setting['env'],
                    str(datetime.now())
                )
            )

        writer.set_as_default()
    logs = Logs(setting['save_dir'])

    preprocess = Preprocess()
    env = gym.make(setting['env'])
    ##----------------------- ----------------------------------------------##
    #[e.id for e in (gym.envs.registry.all()) if "pong" in e.id.lower()]
    ##______________________________________________________________________##

    replay_buffer = RingBuffer(setting['buffer'])
    action_dim = env.action_space.n

    setting['action_dim'] = action_dim
    agent = DDQN(setting)

    agent.take_action = EpsilonGreedy(
                setting['max_epsilon'],
                setting['min_epsilon'],
                setting['step'], 
                setting['fraction'], 
                lambda :[env.action_space.sample()],
                lambda s: agent._take_action(s)
            ).action

    ep_reward = 0
    tracking = []
    episode = 0

    train_info = None

    state = env.reset()
    state = preprocess.atari(state)
    #state = np.stack([state] * 4, axis=2)
    state = [state] * 4

    step = 0
    for step in trange(setting['step']):
        state_ = np.stack(state, axis=2)
        action, action_info = agent.take_action(state_[np.newaxis, :], step)

        next_state, reward, terminal, _ = env.step(tf.squeeze(action).numpy())

        reward = np.clip(reward, -1, 1)
        ep_reward += reward

        next_state = preprocess.atari(next_state)
        next_state = state[1:] + [next_state]

        sample = (state, action, reward, next_state, terminal)
        replay_buffer.add(sample)

        state = next_state
        if step > setting['start']:
            if step % setting['train_step'] == 0:
                batch = replay_buffer.get_batch(setting['batch'])
                train_info = agent.train(batch)
                pass

            if setting['soft_update']:
                agent.soft_update()
            else:
                if step % setting['update'] == 0:
                    agent.hard_update()

        ##-----------------------  TERMINAL SECTION ----------------------------##
        # TODO: wrap tboard and logging section into function



        # TODO: plot result to file to observe result
        if terminal:
            if setting['tboard']:
                tf.summary.scalar('metrics/epsilon', data=action_info['epsilon'], step=step)
                tf.summary.scalar('metrics/episode', data=episode, step=step)
                tf.summary.scalar('metrics/reward', data=ep_reward, step=step)
            if train_info:
                if setting['tboard']:
                    tf.summary.scalar('metrics/loss', data=train_info['loss'], step=step)
                    tf.summary.scalar('metrics/Q', data=train_info['Q'], step=step)
                logs.loss.append((step, round(float(train_info['loss'].numpy()), 3)))
                logs.Q.append((step, round(float(train_info['Q'].numpy()), 3)))
            logs.reward.append((step, ep_reward))

            state = env.reset()
            state = preprocess.atari(state)
            state = [state] * 4
            episode += 1
            ep_reward = 0
        ##______________________________________________________________________##
        # TODO: save train model

    logs.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--tmp-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--env", default='CartPole-v0')

    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--update", type=int, default=100)
    parser.add_argument("--buffer", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n-run", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--train-step", type=int, default=1)
    parser.add_argument("--start", type=int, default=1000)

    parser.add_argument("--fixed-policy", action='store_true')

    parser.add_argument("--soft-update", action='store_true')
    parser.add_argument("--tau", type=float, default=0.001)

    parser.add_argument("--vi", action='store_true')
    parser.add_argument("--pol", action='store_true')

    parser.add_argument("--huber", action='store_true')
    parser.add_argument("--mse", action='store_true')

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--rms", action='store_true')
    parser.add_argument("--adam", action='store_true')
    parser.add_argument("--sgd", action='store_true')

    parser.add_argument("--max-epsilon", type=float, default=0.8)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--fraction", type=float, default=0.3)

    parser.add_argument("--dqn", action='store_true')
    parser.add_argument("--ddqn", action='store_true')

    parser.add_argument("--tboard", action='store_true')
    parser.add_argument("--name")


    args = parser.parse_args()
    setting = vars(args)

    ##-----------------------CREATE NEW FOLDER FOR ENVIRONMENT -------------##
    if setting['tmp_dir']:
        dir_path = join(setting["tmp_dir"], setting["env"])
        os.makedirs(dir_path, exist_ok=True)
        index = 1 + max([0,] \
                + [int(d) for d in os.listdir(dir_path) \
                if isdir(join(dir_path, d)) and d.isnumeric()])
        setting['save_dir'] = join(dir_path, str(index))
        os.makedirs(setting['save_dir'], exist_ok=True)
    ##______________________________________________________________________##
    print(setting['save_dir'])
    with open(os.path.join(setting['save_dir'], 'setting.json'), 'w') as f:
        f.write(json.dumps(setting))

    for _ in range(setting['n_run']):
        train(setting)
