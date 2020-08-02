import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, losses, optimizers, Sequential
from ..algo import EpsilonGreedy
from tqdm import trange
import pandas as pd
from datetime import datetime
from ..util import allow_gpu_growth, incremental_path, Logs, PrintUtil
from collections import namedtuple, deque
from os.path import join, isdir
import numpy.random as npr
import numpy as np

import gym
import argparse
import sys
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

class RingBuffer(object):
    def __init__(self, N):
        #self.buffer = deque(maxlen=N)
        self.buffer = [None] * N
        self.N = N
        self.last_index = -1
        self.index = -1

    def add(self, sample):
        self.index = (self.index + 1) % self.N
        self.last_index = min(self.last_index + 1, self.N)
        self.buffer[self.index] = sample

    #@tf.function
    def get_batch(self, batch_size):
        select_index = npr.choice(self.last_index, batch_size)
        batch = [self.buffer[i] for i in select_index]
        return batch

class MLP(Model):
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
    def __init__(self, args):
        self.debug = args.debug
        obs_dim, action_dim = args.obs_dim, args.action_dim
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

        self.train_net(tf.random.uniform(shape=[1, obs_dim], dtype=TF_TYPE))
        self.fixed_net(tf.random.uniform(shape=[1, obs_dim], dtype=TF_TYPE))
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

    @tf.function
    def _take_action(self, state):
        Q = self.policy_net(state)
        A = tf.argmax(Q, axis=1)
        return A

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
        #tf.print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        with tf.GradientTape() as tape:
            Q = self.train_net(state)
            Q = tf.gather(Q, action, batch_dims=1)
            Q_policy_next = self.next_policy_net(next_state)
            next_action = tf.argmax(Q_policy_next, axis=1)
            next_action = tf.reshape(next_action, shape=(-1, 1))
            Q_next = self.Q_next_net(next_state)
            V_next = tf.gather(Q_next, next_action, batch_dims=1)
            Q_target = reward + self.discount * tf.multiply(V_next, (1. - done))
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

def evaluation(args, agent):
    env = gym.make(args.env)

    state = env.reset()
    total_reward, ep_reward = 0, 0
    n_episode = 0
    for _ in range(args.eval_step):
        if npr.uniform() < args.eval_epsilon:
            action = env.action_space.sample()
        else:
            state = state.astype(np.float32)
            action = agent._take_action(state.reshape(1, -1))
            action = tf.squeeze(action).numpy()
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        state = next_state
        if done:
            n_episode += 1
            total_reward += ep_reward
            ep_reward = 0
            state = env.reset()
    env.close()
    return total_reward / max(1, n_episode)

     


def train(args, train_index):
    args.log_path = incremental_path(join(args.log_dir, '*.json'))
    logs = Logs(args.log_path)

    env = gym.make(args.env)

    replay_buffer = RingBuffer(args.buffer)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    args.obs_dim = obs_dim
    args.action_dim = action_dim
    agent = DDQN(args)

    agent.take_action = EpsilonGreedy(
            args.max_epsilon,
            args.min_epsilon,
            args.training_step, 
            args.final_exploration_step, 
            lambda :[env.action_space.sample()],
            lambda s: agent._take_action(s)).action

    ep_reward = 0
    tracking = []
    episode = 0
    print_util = PrintUtil(args.epoch_step, args.training_step)

    train_info = None
    done = True
    last_score = 0
    best_eval_score = -1e6
    model_path = None
    for step in range(args.training_step):
        if done:
            state = env.reset()
            episode += 1
            last_score = ep_reward
            ep_reward = 0
        state = state.astype(np.float32)
        action, action_info = agent.take_action(state.reshape(1, -1), step)
        action = tf.squeeze(action).numpy()
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.astype(np.float32)

        ep_reward += reward

        sample = (state, action, reward, next_state, done)
        replay_buffer.add(sample)

        state = next_state
        if step > args.learn_start:
            if step % args.update_freq == 0:
                batch = replay_buffer.get_batch(args.batch)
                train_info = agent.train(batch)

            if args.soft_update:
                agent.soft_update()
            else:
                if step % args.update_target == 0:
                    agent.hard_update()

        ##-----------------------  TERMINAL SECTION ----------------------------##

        if step > args.epoch_step and step % args.epoch_step == 0:
            if train_info:
                logs.loss.append((step, round(float(train_info['loss'].numpy()), 3)))
                logs.Q.append((step, round(float(train_info['Q'].numpy()), 3)))
                logs.train_score.append((step, last_score))
            eval_score = evaluation(args, agent)
            logs.eval_score.append((step, eval_score))
            logs.save()
            #Save best model
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                model_path = join(args.model_dir, '{}.ckpt'.format(step))
                agent.save_model(model_path)
            print_util.epoch_print(step, [
                "Train index: {}".format(train_index),
                "Epsilon: {:.2f}".format(action_info['epsilon']),
                "Best eval score: {:.2f}, Train score:{:.2f}, eval score:{:.2f}"\
                        .format(best_eval_score, last_score, eval_score),
                "Model path:{}".format(model_path)
                ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classical control")
    parser.add_argument("--tmp-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--training-step", type=int, default=1000000)
    parser.add_argument("--epoch-step", type=int, default=25000)
    parser.add_argument("--eval-step", type=int, default=4000)
    parser.add_argument("--update-target", type=int, default=1000)
    parser.add_argument("--update-freq", type=int, default=4)
    parser.add_argument("--buffer", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n-run", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learn-start", type=int, default=10000)

    parser.add_argument("--fixed-policy", action='store_true')

    parser.add_argument("--soft-update", action='store_true')
    parser.add_argument("--tau", type=float, default=0.001)

    parser.add_argument("--vi", action='store_true')
    parser.add_argument("--pol", action='store_true')

    parser.add_argument("--huber", action='store_true')
    parser.add_argument("--mse", action='store_true')

    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--rms", action='store_true')
    parser.add_argument("--adam", action='store_true')
    parser.add_argument("--sgd", action='store_true')

    parser.add_argument("--max-epsilon", type=float, default=1)
    parser.add_argument("--min-epsilon", type=float, default=0.1)
    parser.add_argument("--eval-epsilon", type=float, default=0.05)
    parser.add_argument("--final-exploration-step", type=int, default=100000)

    parser.add_argument("--dqn", action='store_true')
    parser.add_argument("--ddqn", action='store_true')

    parser.add_argument("--name")

    args = parser.parse_args()

    ##-----------------------CREATE NEW FOLDER FOR ENVIRONMENT -------------##
    if args.tmp_dir:
        dir_path = join(args.tmp_dir, args.env)
        os.makedirs(dir_path, exist_ok=True)
        index = 1 + max([0,] \
                + [int(d) for d in os.listdir(dir_path) \
                if isdir(join(dir_path, d)) and d.isnumeric()])
        args.save_dir = join(dir_path, str(index))
        args.model_dir = join(args.save_dir, 'model')
        args.log_dir = join(args.save_dir, 'logs')
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    ##______________________________________________________________________##
    print(args.save_dir)
    with open(os.path.join(args.save_dir, 'setting.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    for train_index in range(args.n_run):
        train(args, train_index)
