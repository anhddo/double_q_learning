# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from ..algo import EpsilonGreedy
from datetime import datetime
from ..util import allow_gpu_growth, incremental_path, Logs, PrintUtil
from ..plot_result import log_plot
from os.path import join, isdir
from .atari.ddqn_mask import DDQN
from .atari.replay_buffer import RingBuffer
from timeit import default_timer as timer
import numpy.random as npr
import gym
import argparse
import sys
import json
import time
from PIL import Image
import numpy as np
allow_gpu_growth()

#Sample = namedtuple('Sample', ('state', 'action', 'reward', 'next_state', 'done'))

#tf.keras.backend.set_floatx('float64')
#TF_TYPE = tf.float64

#tf.keras.backend.set_floatx('float32')
#TF_TYPE = tf.float32

def preprocess(img):
    im = Image.fromarray(img)\
            .convert("L")\
            .resize((84, 84), Image.BILINEAR)
    return np.asarray(im)

def init_env(env, noop_action_index, agent, args):
    img = env.reset()
    img = preprocess(img)
    state = np.stack([img] * args.frame_skip, axis=2)
         
    ##----------------------- ----------------------------------------------##
    #Run a maximum NOOP to create the randomness of the game,
    #The agent may overfit to a fix sequence if the game dont have any randomness.
    noop_max = npr.randint(args.random_start)
    for _ in range(noop_max):
        img, reward, end_episode, info = env.step(noop_action_index)
        img = preprocess(img)
        state = np.concatenate((state[:, :, 1:], img[..., np.newaxis]), axis=2)
    return state

def evaluation(args, agent, noop_action_index):
    
    ep_reward = 0
    env = gym.make(args.env)

    total_score = 0
    terminal = True
    start_time = timer()
    episode_start_time = timer()
    total_time = 0
    n_episode = 0
    for step in range(args.eval_step):
        if terminal:
            episode_time = timer() - episode_start_time 
            episode_start_time = timer()
            state = init_env(env, noop_action_index, agent, args)
            total_score += ep_reward
            total_time += episode_time
            n_episode += 1
            ep_reward = 0
        if npr.uniform() < 0.05:
            action = env.action_space.sample()
        else:
            action = agent._take_action(state[np.newaxis,...]) 
        img, reward, terminal, info = env.step(tf.squeeze(action).numpy())
        img = preprocess(img)
        state = np.concatenate((state[:, :, 1:], img[..., np.newaxis]), axis=2)
        ep_reward += reward
    n_episode = max(1, n_episode)
    return {
            'avg_score': total_score / n_episode,
            'avg_time': total_time / n_episode,
            'eval_time': timer() - start_time
            }



def record(args):
    env = gym.make(args.env)
    env = gym.wrappers.Monitor(env, args.record_path, force=True)
    args.action_dim = env.action_space.n
    img = env.reset()
    img = preprocess(img)
    state = np.stack([img] * 4, axis=2)
    agent = DDQN(args)
    agent.load_model(args.load_model_path)
    agent.take_action = EpsilonGreedy(
                0.05,
                0.05,
                1, 
                1, 
                lambda :[env.action_space.sample()],
                lambda s: agent._take_action(s)
            ).action

    while True:
        action, action_info = agent.take_action(state[np.newaxis,...], 0)
        img, reward, end_episode, info = env.step(tf.squeeze(action).numpy())
        img = preprocess(img)
        state = np.concatenate((state[:, :, 1:], img[..., np.newaxis]), axis=2)
        if end_episode:
            break
    env.close()



def train(args):
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
                args.init_exploration,
                args.final_exploration,
                args.training_step, 
                args.final_exploration_step, 
                lambda :[env.action_space.sample()],
                lambda s: agent._take_action(s)
            ).action

    agent.load_model(args.load_model_path)
    agent.hard_update()
    ep_reward = 0
    tracking = []
    episode = 0

    train_info = None
    model_path = None

    last_ep_reward = 0

    print_util = PrintUtil(args.epoch_step, args.training_step)

    action_index = {key: i for i, key in enumerate(env.unwrapped.get_action_meanings())}
    noop_action_index = action_index['NOOP']

    end_episode = True
    ep_reward = 0
    train_info = None
    update_step = 0
    best_score = -1e6
    for step in range(args.training_step):
        ##-----------------------  TERMINAL SECTION ----------------------------##
        if end_episode:
            current_lives = env.unwrapped.ale.lives()
            state = init_env(env, noop_action_index, agent, args)
            state_list = [state[:, :, i] for i in range(args.frame_skip)]
            last_ep_reward = ep_reward
            episode += 1
            ep_reward = 0

        ##______________________________________________________________________##

        if step > args.learn_start and step % args.epoch_step == 0:
            ##-------------------- EVALUATION SECTION ----------------------------------##
            eval_info = evaluation(args, agent, noop_action_index)
            avg_score = eval_info['avg_score']
            eval_time_per_episode = eval_info['avg_time']
            ##----------------------- SAVE MODEL---------------------------##
            if avg_score > best_score:
                best_score = avg_score
                model_path = join(args.model_dir, '{}.ckpt'.format(step))
                agent.save_model(model_path)
            best_score = max(best_score, avg_score)
            ##----------------------- PRINT SECTION---------------------------##
            print_util.epoch_print(step, [
                "Avg eval score: {:.2f}, total: {:.2f} min,  {:.2f} (s/episode)"\
                        .format(avg_score, eval_info['eval_time'] / 60, eval_time_per_episode),
                        "Epsilon: {:.2f}".format(action_info['epsilon']), 
                        "Best score: {:.2f}".format(best_score),
                        args.save_dir,
                        "Model path: {}".format(model_path),
                ])
            logs.eval_reward.append((step, avg_score))
            if train_info:
                logs.loss.append((step, round(float(train_info['loss'].numpy()), 3)))
                logs.Q.append((step, round(float(train_info['Q'].numpy()), 3)))
            logs.train_reward.append((step, last_ep_reward))
            logs.save()
            ##______________________________________________________________________##
        if step > args.learn_start:
            if step % args.update_freq == 0:
                batch = replay_buffer.get_batch()
                train_info = agent.train(batch)

            if args.soft_update:
                agent.soft_update()
            else:
                if step % args.update_target == 0:
                    agent.hard_update()

                    
        #Run an epoch
        if step < args.learn_start:
            action = [env.action_space.sample()]
        else:
            action, action_info = agent.take_action(state[np.newaxis,...], step)
        img, reward, end_episode, info = env.step(tf.squeeze(action).numpy())
        # We set terminal flag is true every time agent loses life
        end_episode = end_episode or info['ale.lives'] < current_lives
        current_lives = info['ale.lives']

        reward = np.clip(reward, -1, 1)
        reward = float(reward)
        ep_reward += reward

        # Stacking the reward to create new next state
        img = preprocess(img)
        state_list.append(img)
        state = np.concatenate((state[:, :, 1:], img[..., np.newaxis]), axis=2)
        replay_buffer.add(state_list, action, reward, float(end_episode))

        state_list = state_list[1:]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--tmp-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--record-path")
    parser.add_argument("--load-model-path")
    parser.add_argument("--env", default='CartPole-v0')

    parser.add_argument("--buffer", type=int, default=1000000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--n-run", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)

    parser.add_argument("--update-freq", type=int, default=4)
    parser.add_argument("--update-target", type=int, default=10000)
    parser.add_argument("--training-step", type=int, default=200000000)
    parser.add_argument("--epoch-step", type=int, default=250000)
    parser.add_argument("--eval-step", type=int, default=135000)
    parser.add_argument("--learn-start", type=int, default=50000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--random-start", type=int, default=30)

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

    parser.add_argument("--init-exploration", type=float, default=1.)
    parser.add_argument("--final-exploration", type=float, default=0.1)
    parser.add_argument("--final-exploration-step", type=int, default=1000000)

    parser.add_argument("--dqn", action='store_true')
    parser.add_argument("--ddqn", action='store_true')

    parser.add_argument("--name")

    args = parser.parse_args()


    if args.debug:
        tf.config.experimental_run_functions_eagerly(True)
    if args.record:
        record(args)
    else:
        ##-----------------------TRAIN SECTION -------------##
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
