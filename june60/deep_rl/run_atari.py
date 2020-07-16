# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import tensorflow as tf
from ..algo import EpsilonGreedy
from tqdm import trange, tqdm
from datetime import datetime
from ..util import allow_gpu_growth, incremental_path, Logs
from ..plot_result import log_plot
from os.path import join, isdir
from .atari.ddqn import DDQN
from .atari.replay_buffer import RingBuffer
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

#Sample = namedtuple('Sample', ('state', 'action', 'reward', 'next_state', 'done'))
#tf.config.experimental_run_functions_eagerly(True)

#tf.keras.backend.set_floatx('float64')
#TF_TYPE = tf.float64

#tf.keras.backend.set_floatx('float32')
#TF_TYPE = tf.float32

def preprocess(img):
    im = Image.fromarray(img)\
            .convert("L")\
            .resize((84, 84), Image.NEAREST)
            #.crop((0, 18, 84, 102))
    return np.asarray(im)

def evaluation(args, agent):
    ep_reward = 0
    env = gym.make(args.env)
    #env = gym.wrappers.Monitor(env, 'tmp/recording', force=True)

    img = env.reset()
    img = preprocess(img)
    state = np.stack([img] * 4, axis=2)
    for _ in range(100000):
        if npr.uniform() < 0.05:
            action = env.action_space.sample()
        else:
            action = agent._take_action(state[np.newaxis,...]) 
        img, reward, terminal, info = env.step(tf.squeeze(action).numpy())
        img = preprocess(img)
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
    current_lives = env.unwrapped.ale.lives()
    img = preprocess(img)
    state_indices = [replay_buffer.insert_image(img) for _ in range(4)]
    state = np.stack([img] * 4, axis=2)

    save_model_step = args.step // args.n_model_save
    save_log_step = args.step // args.n_log_save

    for step in trange(args.step):
        action, action_info = agent.take_action(state[np.newaxis,...], step)
        img, reward, end_episode, info = env.step(tf.squeeze(action).numpy())
        # We set terminal flag is true every time agent loses life
        is_live_loss = info['ale.lives'] < current_lives
        current_lives = info['ale.lives']
        terminal = end_episode or is_live_loss

        reward = np.clip(reward, -1, 1)
        reward = float(reward)
        ep_reward += reward

        # Stacking the reward to create new next state
        img = preprocess(img)
        state_indices.append(replay_buffer.insert_image(img))
        state = np.concatenate((state[:, :, 1:], img[..., np.newaxis]), axis=2)
        replay_buffer.add(state_indices, action, reward, float(terminal))
        state_indices = state_indices[1:]

        if step % args.eval_step == 0:
            eval_reward = evaluation(args, agent)
            logs.eval_reward.append((step, eval_reward))
        train_info = None
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
        if end_episode:
            if args.tboard:
                tf.summary.scalar('metrics/epsilon', data=action_info['epsilon'], step=step)
                tf.summary.scalar('metrics/episode', data=episode, step=step)
                tf.summary.scalar('metrics/reward', data=ep_reward, step=step)
            logs.train_reward.append((step, ep_reward))

            img = env.reset()
            current_lives = env.unwrapped.ale.lives()
            img = preprocess(img)
            state_indices = [replay_buffer.insert_image(img) for _ in range(4)]
            state = np.stack([img] * 4, axis=2)
            episode += 1
            ep_reward = 0

        if train_info and step % save_log_step == 0:
            if args.tboard:
                tf.summary.scalar('metrics/loss', data=train_info['loss'], step=step)
                tf.summary.scalar('metrics/Q', data=train_info['Q'], step=step)
            logs.loss.append((step, round(float(train_info['loss'].numpy()), 3)))
            logs.Q.append((step, round(float(train_info['Q'].numpy()), 3)))
            logs.save()
        ##----------------------- SAVE MODEL---------------------------##
        if step % save_model_step == 0:
            agent.save_model(join(args.model_dir, '{}.ckpt'.format(step // save_model_step)))
            #log_plot(logs.log_path)
        ##______________________________________________________________________##

    logs.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--tmp-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--load-model-path")
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--n-model-save", type=int, default=1000)
    parser.add_argument("--n-log-save", type=int, default=1000000)

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
