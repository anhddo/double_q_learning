import tensorflow as tf
import json
import glob
import os

def allow_gpu_growth():
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

class Logs(object):
    def __init__(self, file_path):
        self.loss = []
        self.reward = []
        self.Q = []
        self.episode = []
        self.path = file_path

    def load(self):
        with open(self.path) as f:
            self.__dict__.update(json.load(f))


    def save(self):
        files = glob.glob(os.path.join(self.path,"*.json"))
        file_names = [os.path.basename(file).split(".")[0] for file in files]
        index = 1 + max([int(e) for e in file_names if e.isnumeric()] + [0])
        file_path = os.path.join(self.path, "{}.json".format(index))
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f)
