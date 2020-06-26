from .ovi import OVI, FourierBasis
from .algo import EpsilonGreedy
import tensorflow as tf
from tqdm import trange
import pandas as pd
from datetime import datetime
from util import allow_gpu_growth
import gym
import argparse
import sys
import os
import json
import glob
import time

allow_gpu_growth()

def train(setting):
    env = gym.make(setting['env_name'])

    if env.spec._env_name == 'CartPole':
        min_clip, max_clip = [0, 500]
    elif env.spec._env_name == 'MountainCar':
        min_clip, max_clip = [-200, 0]
    elif env.spec._env_name == 'Acrobot':
        min_clip, max_clip = [-500, 0]

    A = env.action_space.n

    writer = tf.summary.create_file_writer('{}/logs/{}-{}'\
            .format(os.path.expanduser('~'),
                setting['env_name'],
                str(datetime.now())
            )
        )
    writer.set_as_default()

    state = env.reset()

    fourier_basis = FourierBasis(setting['fourier_order'], env)
    state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
    d = len(state)

    ep_reward = 0
    agent = OVI(A, setting['buffer'], d, min_clip, max_clip, env, setting)

    tracking = []
    agent.take_action = EpsilonGreedy(
            setting['max_epsilon'], 
            setting['min_epsilon'],
            setting['step'],
            setting['fraction'],
            env.action_space.sample,
            lambda state: agent._take_action(state).numpy()).action

    for t in trange(setting['step']):
        action, _ = agent.take_action(state)

        next_state, reward, done, _ = env.step(action)
        ep_reward += reward

        reward = tf.constant(reward, dtype=tf.dtypes.double)
        done = tf.constant(done, dtype=tf.dtypes.double)

        next_state = fourier_basis.transform(tf.constant(next_state, dtype=tf.dtypes.double))


        agent.update_inverse_covariance(action, state)
        agent.observe(state, action, reward, next_state, done)
        if t % setting['sample_len'] == 0:
            agent.train()

        state = next_state

        if done:
            tf.summary.scalar('metrics/reward', data=ep_reward, step=t)
            tracking.append([ep_reward, t])
            state = env.reset()
            state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
            ep_reward = 0

    files = glob.glob(os.path.join(setting['save_dir'],"*.csv"))
    csv_index = 1 + max([int(os.path.basename(file).split('.')[0]) for file in files] + [0])
    csv_path = os.path.join(setting['save_dir'], '{}.csv'.format(csv_index))
    reward, timestep = zip(*tracking)
    df = pd.DataFrame({'t': timestep, 'reward': reward, 'algo': setting['algo']})
    df.to_csv(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--fourier-order", type=int, default=1)
    parser.add_argument("--env-name", default='CartPole-v0')
    parser.add_argument("--algo", default='vi')
    parser.add_argument("--buffer", type=int, default=5000)
    parser.add_argument("--save-dir")
    parser.add_argument("--n-run", type=int, default=10)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--sample-len", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--max-epsilon", type=float, default=0.8)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--fraction", type=float, default=0.3)

    args = parser.parse_args()
#     args = parser.parse_args('--fourier-order 2 --env-name Acrobot-v1 --save-dir tmp/bot --buffer-size 2000 --step 20000 --beta 1  --n-run 5'.split())

    setting = vars(args)

    os.makedirs(setting['save_dir'], exist_ok=True)
    with open(os.path.join(setting['save_dir'], 'setting.json'), 'w') as f:
        f.write(json.dumps(setting))

    for _ in range(setting['n_run']):
        train(setting)
        #time.sleep(1000)
