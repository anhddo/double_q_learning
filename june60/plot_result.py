# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import join
import argparse
from os import listdir
from .util import Logs
from tqdm import tqdm
import os
"""
Structure of a folder:
    BreakoutDeterministic-v4
    |---settings.json
    |---model                //Save best model
    |---|---1.ckpt
    |---|---2.ckpt
    |---logs                 //Save all the logs for plotting
    |---|---1.json
    |---|---2.json
"""

def log_plot(args):
    log_file_path = args.log_path
    with open(log_file_path) as f:
        dir_path, file_name = os.path.split(log_file_path)
        index, _ = file_name.split('.')
        logs = json.load(f)
        fig = plt.gcf()
        fig.set_size_inches(args.width, args.height)
        plt.clf()

        plt.subplot(311)
        plt.grid()
        x, y = zip(*logs['train_score'])
        plt.plot(x, y, label='train score')
        x, y = zip(*logs['eval_score'])
        plt.plot(x, y, label='eval score')
        plt.xlabel('frames')
        plt.legend()

        plt.subplot(312)
        plt.grid()
        x, y = zip(*logs['loss'])
        plt.plot(x, y, label='loss')
        plt.xlabel('frames')
        plt.legend()

        plt.subplot(313)
        plt.grid()
        plt.grid()
        x, y = zip(*logs['Q'])
        plt.plot(x, y, label='Q value')
        plt.xlabel('frames')
        plt.legend()
        plt.savefig(join(dir_path, '{}.pdf'.format(index)))

def get_info(save_dir):
    with open(join(save_dir, 'setting.json')) as f:
        setting = json.loads(f.read())
        x = range(setting['training_step'])
        train_score, eval_score, loss = [], [], []
        logs_dir = join(save_dir, 'logs')
        files = [e for e in listdir(logs_dir) if e.split(".")[0].isnumeric()]
        files = [join(logs_dir, e) for e in files]
        print(files)
        for file_name in files:
            logs = Logs(file_name)
            logs.load()

            timestep, y = zip(*logs.train_score)
            train_score.append(np.interp(x, timestep, y))

            timestep, y = zip(*logs.eval_score)
            eval_score.append(np.interp(x, timestep, y))

        train_score, eval_score = np.stack(train_score), np.stack(eval_score)
        return {'train_score': train_score, 'eval_score': eval_score, 'setting': setting}

def fill_plot(y, label):
    m = np.mean(y, axis=0)
    s = np.std(y, axis=0)
    step = len(m)
    plt.plot(m, label=label, linewidth=1, alpha=0.8)
    plt.fill_between(range(step), m - s, m + s, alpha=0.1)

def avg_plot(args):
    plt.grid()
    plt.gcf().set_size_inches(args.width, args.height)

    ##-------------------Get the result of different algorithm -------------##
    for save_dir_, plot_label in zip(args.save_dir, args.label):
        log_info = get_info(save_dir_)
        fill_plot(log_info['train_score'], plot_label + '-train_score')
        fill_plot(log_info['eval_score'], plot_label + '-eval_score')
    ##______________________________________________________________________##
    plt.legend()

    plt.savefig(args.plot_name)

def line_plot(args):
    log_info = get_info(args)
    reward = log_info['reward']
    plt.grid()
    plt.gcf().set_size_inches(args.width, args.height)
    for r in reward:
        plt.plot(r)
    plt.savefig(args.plot_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--save-dir", nargs='+' )
    parser.add_argument("--log-path")
    parser.add_argument("--label", nargs='+')
    parser.add_argument("--plot-name")
    parser.add_argument("--line", action='store_true')
    parser.add_argument("--avg", action='store_true')
    parser.add_argument("--width", type=float, default=20)
    parser.add_argument("--height", type=float, default=5)
    args = parser.parse_args()
    #args = vars(args)


    if args.log_path:
        log_plot(args)
    if args.line:
        line_plot(args)
    elif args.avg:
        avg_plot(args)
