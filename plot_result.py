import numpy as np
import numpy.random as npr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json
from os import path

def avg_plot(dir_path):
    with open(path.join(dir_path, 'setting.json')) as f:
        setting = (json.loads(f.read()))
        x = range(setting['step'])
        X, Y = [], []
        files = glob.glob(path.join(dir_path, "*.csv"))
        for file in files:
            df = pd.read_csv(file)
            X.append(list(x))
            Y.append(np.interp(x, df.t, df.reward))

        X, Y = np.stack(X), np.stack(Y)

        m = np.mean(Y, axis=0)
        s = np.std(Y, axis=0)
        plt.plot(m, label=setting['algo'], color='r', linewidth=1, alpha=0.8)
        plt.fill_between(range(setting['step']), m - s, m + s, alpha=0.1)

