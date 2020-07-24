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
    def __init__(self, args):
        self.start_time = timer()
        self.last_eval_time = self.start_time
        self.it_each_epoch = args.frame_each_epoch // args.frame_skip
        self.training_frame = args.training_frame


    def calc_date_time(self, second):
        day = int(second / (24 * 3600))
        hour = int((second - day * 24 * 3600) / 3600)
        minute = int((second - day * 24 * 3600 - hour * 3600) / 60)
        return day, hour, minute

    def epoch_print(self, frame, pstr):
        time_elapsed = timer() - self.start_time
        speed = int(self.it_each_epoch / (timer() - self.last_eval_time))
        time_left = (self.training_frame - frame) / speed
        day, hour, minute = self.calc_date_time(time_elapsed)
        day_left, hour_left, minute_left = self.calc_date_time(time_left)

        pstr = [
                "Frame: {:.2f}e6/{}e6,{:2d}%, Speed:{} it/s, Epoch time: {:.2f}min" 
                .format(frame / 1e6,\
                        self.training_frame // 1000000, \
                        int(frame / self.training_frame * 100), \
                        speed, \
                        (timer() - self.last_eval_time) / 60),\
                "Elapsed time:{}d-{}h-{}m, Time left: {}d-{}h-{}m"\
                .format(day, hour, minute, day_left, hour_left, minute_left),
                ] + pstr

        self.last_eval_time = timer()

        L = max([len(e) for e in pstr])
        print('|'+'=' * L+'|')
        for e in pstr:
            print('|' + e + ' ' * (L - len(e)) + '|')
        print('|'+'=' * L+'|')
        print('\n')

