import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from .ovi import OVI, FourierBasis
from ..algo import EpsilonGreedy
import tensorflow as tf
from tqdm import trange
import pandas as pd
from datetime import datetime
from ..util import allow_gpu_growth, incremental_path, Logs, PrintUtil, make_training_dir, save_setting
from os.path import join
import gym
import argparse
import sys
import os
import json
import glob
import time

allow_gpu_growth()

def train(args):
    args.log_path = incremental_path(join(args.log_dir, '*.json'))
    print(args.log_path)
    logs = Logs(args.log_path)
    env = gym.make(args.env)

    if env.spec._env_name == 'CartPole':
        min_clip, max_clip = [0, 500]
    elif env.spec._env_name == 'MountainCar':
        min_clip, max_clip = [-200, 0]
    elif env.spec._env_name == 'Acrobot':
        min_clip, max_clip = [-500, 0]

    A = env.action_space.n

    if args.debug:
        writer = tf.summary.create_file_writer('{}/logs/{}-{}'\
                .format(os.path.expanduser('~'),
                    args.env_name,
                    str(datetime.now())
                )
            )
        writer.set_as_default()

    state = env.reset()

    fourier_basis = FourierBasis(args.fourier_order, env)
    state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
    d = len(state)

    ep_reward = 0
    agent = OVI(A, args.buffer, d, min_clip, max_clip, args.beta, env)

    tracking = []
    agent.take_action = EpsilonGreedy(
            args.max_epsilon, 
            args.min_epsilon,
            args.training_step,
            args.final_exploration_step,
            env.action_space.sample,
            lambda state: agent._take_action(state).numpy()).action

    print_util = PrintUtil(args.epoch_step, args.training_step)
    for t in range(args.training_step):
        action, _ = agent.take_action(state, t)

        next_state, reward, done, _ = env.step(action)
        ep_reward += reward

        reward = tf.constant(reward, dtype=tf.dtypes.double)
        done = tf.constant(done, dtype=tf.dtypes.double)

        next_state = fourier_basis.transform(tf.constant(next_state, dtype=tf.dtypes.double))


        agent.update_inverse_covariance(action, state)
        agent.observe(state, action, reward, next_state, done)
        if t % args.train_freq == 0:
            agent.train()

        state = next_state
        if t % args.epoch_step == 0:
            print_util.epoch_print(t, [
                "Last rewards:{}".format(logs.train_score[-5:])
                ])

        if done:
            logs.train_score.append((t, ep_reward))
            state = env.reset()
            state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
            ep_reward = 0

    if args.write_result:
        logs.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--fourier-order", type=int, default=1)
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--algo", default='vi')
    parser.add_argument("--buffer", type=int, default=5000)
    parser.add_argument("--tmp-dir", default='~/tmp')
    parser.add_argument("--n-run", type=int, default=50)
    parser.add_argument("--training-step", type=int, default=25000)
    parser.add_argument("--epoch-step", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--max-epsilon", type=float, default=0.8)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--final-exploration-step", type=float, default=5000)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--write-result", action='store_true')

    args = parser.parse_args()


    make_training_dir(args)
    save_setting(args)
    print(args.save_dir)

    for _ in range(args.n_run):
        train(args)
        if args.pause:
            time.sleep(1000)
