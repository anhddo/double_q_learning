import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import join
import argparse
from os import listdir
from .util import Logs
from tqdm import tqdm
import os

def log_plot(args):
    log_file_path = args.log_path
    with open(log_file_path) as f:
        dir_path, file_name = os.path.split(log_file_path)
        index, _ = file_name.split('.')
        logs = json.load(f)
        for tag in ['loss', 'train_reward', 'eval_reward']:
            if logs[tag]:
                tag_png = join(dir_path, '{}_{}.pdf'.format(index, tag))
                x, y = zip(*logs[tag])
                fig = plt.gcf()
                fig.set_size_inches(args.width, args.height)
                plt.clf()
                plt.plot(x, y)
                plt.savefig(tag_png)

def get_info(logs_path):
    with open(join(logs_path, 'setting.json')) as f:
        setting = (json.loads(f.read()))
        x = range(setting['step'])
        reward, loss = [], []
        files = [e for e in listdir(logs_path) if e.split(".")[0].isnumeric()]
        files = [join(logs_path, e) for e in files]
        for file_name in tqdm(files):
            logs = Logs(file_name)
            logs.load()
            timestep, y = zip(*logs.reward)
            reward.append(np.interp(x, timestep, y))
            timestep, y = zip(*logs.reward)
            loss.append(np.interp(x, timestep, y))

        reward, loss = np.stack(reward), np.stack(loss)
        return {'reward': reward, 'loss': loss, 'setting': setting}

def avg_plot(args):
    plt.grid()
    plt.gcf().set_size_inches(args['width'], args['height'])

    ##-------------------Get the result of different algorithm -------------##
    for file_path, plot_label in zip(args['log_dir'], args['label']):
        log_info = get_info(file_path)
        m = np.mean(log_info['reward'], axis=0)
        s = np.std(log_info['reward'], axis=0)
        plt.plot(m, label=plot_label, linewidth=1, alpha=0.8)
        plt.fill_between(range(log_info['setting']['step']), m - s, m + s, alpha=0.1)
    ##______________________________________________________________________##
    plt.legend()

    plt.savefig(args['plot_name'])

def line_plot(args):
    log_info = get_info(args)
    reward = log_info['reward']
    plt.grid()
    plt.gcf().set_size_inches(args['width'], args['height'])
    for r in reward:
        plt.plot(r)
    plt.savefig(args['plot_name'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finite-horizon MDP")
    parser.add_argument("--log-dir", nargs='+' )
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
