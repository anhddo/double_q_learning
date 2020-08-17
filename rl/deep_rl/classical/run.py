import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
from datetime import datetime
from collections import namedtuple, deque
from os.path import join, isdir
import numpy.random as npr
import numpy as np
import gym , argparse , sys , json , glob , time

from rl.util import allow_gpu_growth, incremental_path, Logs, PrintUtil
from rl.algo import EpsilonGreedy
from .model import DDQN
from .replay_buffer import RingBuffer


allow_gpu_growth()

Sample = namedtuple('Sample', ('state', 'action', 'reward', 'next_state', 'done'))

#tf.config.experimental_run_functions_eagerly(True)

#tf.keras.backend.set_floatx('float64')
#TF_TYPE = tf.float64

#tf.keras.backend.set_floatx('float32')


def evaluation(args, agent):
    env = gym.make(args.env)

    state = env.reset()
    total_reward, ep_reward = 0, 0
    n_episode = 0
    for _ in range(args.eval_step):
        state = state.astype(np.float32)
        action = agent.take_action_eval(state.reshape(1, -1))
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


    ep_reward = 0
    tracking = []
    episode = 0
    print_util = PrintUtil(args.epoch_step, args.training_step)

    init_score = evaluation(args, agent)
    logs.train_score.append((0, init_score))
    logs.eval_score.append((0, init_score))

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
        action, action_info = agent.take_action_train(state.reshape(1, -1), step)
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
    parser.add_argument("--optimistic", action='store_true')


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
