import tensorflow as tf
import json
import glob
import os
from os.path import join, isdir
from timeit import default_timer as timer

def allow_gpu_growth():
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
##----------------------- ----------------------------------------------##
"""
"""
def incremental_path(path_pattern, is_dir=False):
    if is_dir:
        index = 1 + max([0,] \
                + [int(d) for d in os.listdir(path_pattern) \
                if isdir(join(path_pattern, d)) and d.isnumeric()])
        return join(path_pattern, str(index))
    else:
        files = glob.glob(path_pattern)
        file_names = [os.path.basename(file).split(".")[0] for file in files]
        index = 1 + max([int(e) for e in file_names if e.isnumeric()] + [0])
        dir_path, file_name = os.path.split(path_pattern)
        file_name_str, extension = file_name.split('.')
        file_path = os.path.join(dir_path, "{}.{}".format(index, extension))
        return file_path

class Logs(object):
    def __init__(self, file_path):
        self.loss = []
        self.train_reward = []
        self.eval_reward = []
        self.Q = []
        self.episode = []
        self.log_path = file_path

    def load(self):
        with open(self.path) as f:
            self.__dict__.update(json.load(f))

    def save(self):
        #file_path = incremental_path(os.path.join(self.path,"*.json"))
        with open(self.log_path, 'w') as f:
            json.dump(self.__dict__, f)

class PrintUtil():
    def __init__(self, eval_step, total_step):
        self.start_time = timer()
        self.last_eval_time = self.start_time
        self.eval_step = eval_step
        self.total_step = total_step


    def calc_date_time(self, second):
        day = int(second / (24 * 3600))
        hour = int((second - day * 24 * 3600) / 3600)
        minute = int((second - day * 24 * 3600 - hour * 3600) / 60)
        return day, hour, minute

    def eval_print(self, step, pstr):
        time_elapsed = timer() - self.start_time
        speed = int(self.eval_step / (timer() - self.last_eval_time))
        self.last_eval_time = timer()
        time_left = (self.total_step - step) / speed
        day, hour, minute = self.calc_date_time(time_elapsed)
        day_left, hour_left, minute_left = self.calc_date_time(time_left)

        pstr = [
                "{:2d}%, speed:{} it/s, elapsed time:{}d-{}h-{}m, time left: {}d-{}h-{}m"
                .format(int(step / self.total_step * 100),
                        speed, day, hour, minute,
                        day_left, hour_left, minute_left),
                ] + pstr

        L = max([len(e) for e in pstr])
        print('|'+'=' * L+'|')
        for e in pstr:
            print('|' + e + ' ' * (L - len(e)) + '|')
        print('|'+'=' * L+'|')
        print('\n')

