# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model, losses, optimizers, Sequential
from ..algo import EpsilonGreedy
from tqdm import trange, tqdm
from datetime import datetime
from ..util import allow_gpu_growth, incremental_path, Logs
from ..plot_result import log_plot
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
    def __init__(self, N, batch_size):
        self.buffer = [None] * N
        self.N = N
        self.last_index = -1 
        self.index = -1 

        img_buf_size = int(N * 1.5)
        self.free_indices = deque(range(img_buf_size))
        self.image_buf = np.zeros((img_buf_size, 84, 84), dtype=np.uint8)
        self.state_batch = np.zeros((batch_size, 84, 84, 4), dtype=np.uint8)
        self.next_state_batch = np.zeros((batch_size, 84, 84, 4), dtype=np.uint8)
        self.batch_size = batch_size

    def add(self, state_indices, action, reward, done):
        self.buffer.append((state_indices, action, reward, done))
        self.index = (self.index + 1) % self.N
        self.last_index = min(self.last_index + 1, self.N)
        if self.buffer[self.index] is not None:
            removed_indices, _, _, _ = self.buffer[self.index]
            for e in removed_indices:
                self.free_indices.append(e)
        self.buffer[self.index] = (state_indices, action, reward, done)

    def insert_image(self, img):
        ind = self.free_indices.pop()
        self.image_buf[ind, ...] = img
        return ind

    def get_batch(self):
        select_index = npr.choice(self.last_index, self.batch_size)
        state_indices, action, reward, done = zip(*[self.buffer[i] for i in select_index])
        for i, indices in enumerate(state_indices):
            self.state_batch[i, ...] = self.image_buf[indices[:-1]].transpose((1, 2, 0))
            self.next_state_batch[i, ...] = self.image_buf[indices[1:]].transpose((1, 2, 0))
        action = np.stack(action)
        reward = np.array(reward, dtype=np.float32).reshape(self.batch_size, 1)
        done = np.array(done, dtype=np.float32).reshape(self.batch_size, 1)
        return self.state_batch, action, reward, self.next_state_batch, done


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
        x = tf.cast(x, TF_TYPE) / 255.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class DDQN(object):
    def __init__(self, args):
        action_dim = args.action_dim
        self.discount = args.discount
        self.batch_size = args.batch

        self.use_huber = args.huber
        self.use_mse = args.mse


        self.use_rms = args.rms
        self.use_adam = args.adam

        self.tau = args.tau
        self.tboard = args.tboard

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

        self.train_net(tf.random.uniform(shape=[1, 84, 84, 4], dtype=TF_TYPE))
        self.fixed_net(tf.random.uniform(shape=[1, 84, 84, 4], dtype=TF_TYPE))
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

            loss = self.loss_func(Q, Q_target)
        grad = tape.gradient(loss, self.train_net.trainable_variables)
        #grad = [tf.clip_by_value(e, -1., 1.) for e in grad]
        self.optimizer.apply_gradients(zip(grad, self.train_net.trainable_variables))
        return {
                'loss': loss,
                'Q': tf.reduce_max(Q),
                'Q_target': tf.reduce_max(Q_target)
                }

class Preprocess(object):
    def atari(self, img):
        im = Image.fromarray(img)\
                .convert("L")\
                .resize((84, 84), Image.NEAREST)
                #.crop((0, 18, 84, 102))
        return np.asarray(im)

def evaluation(args, agent):
    ep_reward = 0
    env = gym.make(args.env)
    #env = gym.wrappers.Monitor(env, 'tmp/recording', force=True)

    preprocess = Preprocess()
    img = env.reset()
    img = preprocess.atari(img)
    state = np.stack([img] * 4, axis=2)
    for _ in range(100000):
        if npr.uniform() < 0.05:
            action = env.action_space.sample()
        else:
            action = agent._take_action(state[np.newaxis,...]) 
        img, reward, terminal, info = env.step(tf.squeeze(action).numpy())
        img = preprocess.atari(img)
        state = np.concatenate((state[:, :, 1:], img[..., np.newaxis]), axis=2)
        ep_reward += reward
        if terminal:
            return ep_reward
    print("Evaluate over 100k frames")


def train(args):
    if args.tboard:
        writer = tf.summary.create_file_writer('{}/logs/{}-{}'\
                .format(os.path.expanduser('~'),
                    args.env,
                    str(datetime.now())
                )
            )

        writer.set_as_default()
    args.log_path = incremental_path(join(args.log_dir, '*.json'))
    logs = Logs(args.log_path)

    preprocess = Preprocess()
    env = gym.make(args.env)
    ##----------------------- ----------------------------------------------##
    #[e.id for e in (gym.envs.registry.all()) if "pong" in e.id.lower()]
    ##______________________________________________________________________##

    replay_buffer = RingBuffer(args.buffer, args.batch)
    action_dim = env.action_space.n

    args.action_dim = action_dim
    agent = DDQN(args)

    agent.take_action = EpsilonGreedy(
                args.max_epsilon,
                args.min_epsilon,
                args.step, 
                args.fraction, 
                lambda :[env.action_space.sample()],
                lambda s: agent._take_action(s)
            ).action

    agent.load_model(args.load_model_path)
    ep_reward = 0
    tracking = []
    episode = 0

    train_info = None

    img = env.reset()
    img = preprocess.atari(img)
    state_indices = [replay_buffer.insert_image(img) for _ in range(4)]
    state = np.stack([img] * 4, axis=2)

    step = 0

    save_step = args.step // args.n_save
    init_lives = env.unwrapped.ale.lives()
    
    save_train_step = int(step // 1e6) + 1

    for step in trange(args.step):
        action, action_info = agent.take_action(state[np.newaxis,...], step)
        img, reward, terminal, info = env.step(tf.squeeze(action).numpy())
        is_live_loss = info['ale.lives'] < init_lives
        terminal = terminal or is_live_loss

        reward = np.clip(reward, -1, 1)
        reward = float(reward)
        ep_reward += reward

        img = preprocess.atari(img)
        state_indices.append(replay_buffer.insert_image(img))
        state = np.concatenate((state[:, :, 1:], img[..., np.newaxis]), axis=2)
        replay_buffer.add(state_indices, action, reward, float(terminal))
        state_indices = state_indices[1:]
        #args.eval_step = 10
        if step % args.eval_step == 0:
            eval_reward = evaluation(args, agent)
            logs.eval_reward.append((step, eval_reward))
        if step > args.start:
            if step % args.train_step == 0:
                batch = replay_buffer.get_batch()
                train_info = agent.train(batch)
                pass

            if args.soft_update:
                agent.soft_update()
            else:
                if step % args.update == 0:
                    agent.hard_update()

        ##-----------------------  TERMINAL SECTION ----------------------------##
        if terminal:
            if args.tboard:
                tf.summary.scalar('metrics/epsilon', data=action_info['epsilon'], step=step)
                tf.summary.scalar('metrics/episode', data=episode, step=step)
                tf.summary.scalar('metrics/reward', data=ep_reward, step=step)
            if train_info and step % save_train_step == 0:
                if args.tboard:
                    tf.summary.scalar('metrics/loss', data=train_info['loss'], step=step)
                    tf.summary.scalar('metrics/Q', data=train_info['Q'], step=step)
                logs.loss.append((step, round(float(train_info['loss'].numpy()), 3)))
                logs.Q.append((step, round(float(train_info['Q'].numpy()), 3)))
            logs.train_reward.append((step, ep_reward))

            img = env.reset()
            img = preprocess.atari(img)
            state_indices = [replay_buffer.insert_image(img) for _ in range(4)]
            state = np.stack([img] * 4, axis=2)
            episode += 1
            ep_reward = 0
        ##----------------------- SAVE MODEL AND LOGS---------------------------##
        #save_step = 40
        if step % save_step == 0:
            agent.save_model(join(args.model_dir, '{}.ckpt'.format(int(step // save_step))))
            logs.save()
            #log_plot(logs.log_path)
        ##______________________________________________________________________##

    logs.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--tmp-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--load-model-path")
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--n-save", type=int, default=100)

    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--update", type=int, default=100)
    parser.add_argument("--buffer", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n-run", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--train-step", type=int, default=1)
    parser.add_argument("--eval-step", type=int, default=50000)
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

    ##-----------------------CREATE NEW FOLDER FOR ENVIRONMENT -------------##
    if args.tmp_dir:
        dir_path = join(args.tmp_dir, args.env)
        os.makedirs(dir_path, exist_ok=True)

        args.save_dir = incremental_path(dir_path, is_dir=True)
        args.model_dir = join(args.save_dir, 'model')
        args.log_dir = join(args.save_dir, 'logs')

        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    ##______________________________________________________________________##
    print(args.save_dir)
    with open(os.path.join(args.save_dir, 'setting.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
         

    for _ in range(args.n_run):
        train(args)
