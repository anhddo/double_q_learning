import numpy as np
import numpy.random as npr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json
from os import path
import argparse

def get_reward(args):
    dir_path = args['save_dir']
    with open(path.join(dir_path, 'setting.json')) as f:
        setting = (json.loads(f.read()))
        x = range(setting['step'])
        time_step, reward = [], []
        files = glob.glob(path.join(dir_path, "*.csv"))
        dir_path = setting['save_dir']
        for file in files:
            df = pd.read_csv(file)
            time_step.append(list(x))
            reward.append(np.interp(x, df.t, df.reward))

        time_step, reward = np.stack(time_step), np.stack(reward)
        return {'reward': reward, 'setting': setting}

def avg_plot(args):
    reward_info = get_reward(args)
    m = np.mean(reward_info['reward'], axis=0)
    s = np.std(reward_info['reward'], axis=0)
    setting = reward_info['setting']
    plt.grid()
    plt.gcf().set_size_inches(args['width'], args['height'])
    plt.plot(m, label='dqn', linewidth=1, alpha=0.8)
    plt.fill_between(range(setting['step']), m - s, m + s, alpha=0.1)
    plt.savefig(args['plot_name'])

def line_plot(args):
    reward_info = get_reward(args)
    reward = reward_info['reward']
    plt.grid()
    plt.gcf().set_size_inches(args['width'], args['height'])
    for r in reward:
        plt.plot(r)
    plt.savefig(args['plot_name'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--save-dir")
    parser.add_argument("--plot-name")
    parser.add_argument("--line", action='store_true')
    parser.add_argument("--avg", action='store_true')
    parser.add_argument("--width", type=float, default=10)
    parser.add_argument("--height", type=float, default=5)
    args = parser.parse_args()
    args = vars(args)
    print(args)
    if args['line']:
        line_plot(args)
    elif args['avg']:
        avg_plot(args)
