import tensorflow as tf
from tensorflow.keras.layers import Dense
from ovi.algo import epsilon_greedy
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


class MLP(Model):
    def __init__(self, input_dim, out_dim):
	super(DQN, self).__init__()
	self.fn = Dense(64, input_shape=(input_dim,), activation='relu')
        self.head = Dense(out_dim)

    def call(self, x):
	x = self.fn(x)
	return self.head(x)

class DDQN(object):
    def __init__(self, obs_dim, action_dim):
        self.policy_net = MLP(obs_dim, action_dim)
        self.target_net = MLP(obs_dim, action_dim)

    def update(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def observe(self, s, a, r , s1, done):





def train(setting):
    env = gym.make(setting['env_name'])

    A = env.action_space.n

    writer = tf.summary.create_file_writer('{}/logs/{}-{}'\
            .format(os.path.expanduser('~'),
                setting['env_name'],
                str(datetime.now())
            )
        )

    writer.set_as_default()

    s = env.reset()

    agent.take_action = epsilon_greedy(
            setting['epsilon'],
            env.action_space.sample, 
            lambda s: agent._take_action(s).numpy())


    for t in trange(setting['step']):
        a = agent.take_action(s)

        s1, r, done, _ = env.step(a)
        ep_reward += r

        r = tf.constant(r, dtype=tf.dtypes.double)
        done = tf.constant(done, dtype=tf.dtypes.double)

        agent.observe(s, a, r, s1, done)
        if t % setting['sample_len'] == 0:
            agent.train()

        s = s1

        if done:
            tf.summary.scalar('metrics/reward', data=ep_reward, step=t)
            tracking.append([ep_reward, t])
            s = env.reset()
            s = fourier_basis.transform(tf.constant(s, dtype=tf.dtypes.double))
            ep_reward = 0

    csv_index = 1 + max([int(os.path.basename(file).split('.')[0]) for file in glob.glob(os.path.join(setting['save_dir'],"*.csv"))] + [0])
    csv_path = os.path.join(setting['save_dir'], '{}.csv'.format(csv_index))
    reward, timestep = zip(*tracking)
    df = pd.DataFrame({'t': timestep, 'reward': reward, 'algo': setting['algo']})
    df.to_csv(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--fourier-order", type=int, default=1)
    parser.add_argument("--env-name", default='CartPole-v0')
    parser.add_argument("--algo", default='vi')
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--save-dir")
    parser.add_argument("--n-run", type=int, default=10)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--sample-len", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--epsilon", type=float, default=0)

    args = parser.parse_args()
#     args = parser.parse_args('--fourier-order 2 --env-name Acrobot-v1 --save-dir tmp/bot --buffer-size 2000 --step 20000 --beta 1  --n-run 5'.split())

    setting = vars(args)

    os.makedirs(setting['save_dir'], exist_ok=True)
    with open(os.path.join(setting['save_dir'], 'setting.json'), 'w') as f:
        f.write(json.dumps(setting))

    for _ in range(setting['n_run']):
        train(setting)
        time.sleep(1800)
