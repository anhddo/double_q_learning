import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from .util import FourierBasis, EnvWrapper
from .model import ValueIteration, OptimisticValueIteration
from ..algo import EpsilonGreedy
import tensorflow as tf
from datetime import datetime
from timeit import default_timer as timer
from ..util import allow_gpu_growth, incremental_path, Logs, PrintUtil,\
        make_training_dir, save_setting, date_time_string
from os.path import join
import gym
import argparse
import sys
import os
import json
import glob
import time

allow_gpu_growth()

def eval(agent, env):
    ep_reward = 0
    state = env.reset()
    state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
    done = False
    while not done:
        action = agent.take_action(state)
        action = action.numpy()
        state, reward, done, _ = env.step(action)
        state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
        ep_reward += reward
    return ep_reward




def train(args, train_index, fourier_basis, env):
    args.log_path = incremental_path(join(args.log_dir, '*.json'))
    logs = Logs(args.log_path)

    ep_reward = 0
    if args.std_vi:
        agent = ValueIteration(args)
    elif args.optimistic:
        agent = OptimisticValueIteration(args)


    reward = eval(agent, env)
    logs.train_score.append((0, reward))
    #    agent.take_action = EpsilonGreedy(
    #            args.max_epsilon, 
    #            args.min_epsilon,
    #            args.training_step,
    #            args.final_exploration_step,
    #            env.action_space.sample,
    #            lambda state: agent._take_action(state).numpy()).action

    print_util = PrintUtil(args.epoch_step, args.training_step)
    state = env.reset()
    state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
    for t in range(args.training_step):
        action = agent.take_action_train(state)
        action = action.numpy()
        

        next_state, reward, done, _ = env.step(action)
        ep_reward += reward

        reward = tf.constant(reward, dtype=tf.dtypes.double)
        done = tf.constant(done, dtype=tf.dtypes.double)

        next_state = fourier_basis.transform(tf.constant(next_state, dtype=tf.dtypes.double))

        #agent.update_inverse_covariance(action, state)
        agent.observe(state, action, reward, next_state, done)
        #if t % args.train_freq == 0 and t > 500:
        if t % args.train_freq == 0:
            agent.train()

        state = next_state
        if t % args.epoch_step == 0:
            time_elapsed = timer() - args.start_time
            print_util.epoch_print(t, [
                "Last rewards:{}".format(logs.train_score[-5:]),
                "Train index: {}/{}".format(train_index, args.n_run),
                "Folder path:{}".format(args.save_dir),
                "Total time elapsed:{}".format(date_time_string(time_elapsed))
                ])
            logs.save()

        if done:
            logs.train_score.append((t, ep_reward))
            state = env.reset()
            state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
            ep_reward = 0

    #if args.write_result:
    #    logs.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--fourier-order", type=int, default=1)
    parser.add_argument("--env", default='CartPole-v0')
    parser.add_argument("--buffer", type=int, default=5000)
    parser.add_argument("--tmp-dir", default='~/tmp')
    parser.add_argument("--n-run", type=int, default=50)
    parser.add_argument("--training-step", type=int, default=25000)
    parser.add_argument("--epoch-step", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--max-epsilon", type=float, default=1)
    parser.add_argument("--min-epsilon", type=float, default=0.1)
    parser.add_argument("--final-exploration-step", type=float, default=5000)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--pause", action='store_true')
    parser.add_argument("--std-vi", action='store_true')
    parser.add_argument("--optimistic", action='store_true')
    parser.add_argument("--egreedy", action='store_true')
    parser.add_argument("--write-result", action='store_true')

    args = parser.parse_args()

    make_training_dir(args)
    ##----------------------- augment info into args----------------------------------------------##


    env = gym.make(args.env)
    args.n_action = env.action_space.n
    env = EnvWrapper(env)

    fourier_basis = FourierBasis(args.fourier_order, env)
    #state = fourier_basis.transform(tf.constant(state, dtype=tf.dtypes.double))
    args.ftr_dim = fourier_basis.ftr_dim #len(state)
    args.start_time = timer()
    args.min_clip, args.max_clip = env.min_clip, env.max_clip
    save_setting(args)
    ##----------------------------------------------------------------------##
    for train_index in range(args.n_run):
        train(args, train_index + 1, fourier_basis, env)
        if args.pause:
            time.sleep(1000)
