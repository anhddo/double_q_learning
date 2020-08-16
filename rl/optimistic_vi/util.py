import tensorflow as tf
import numpy as np

class FourierBasis(object):
    """
    We featurize the given observation using
    Value Function Approximation in Reinforcement Learning using the Fourier Basis, 
    George Konidaris, etal
    """
    def __init__(self, fourier_order, env):
        tf_type = tf.dtypes.double
        if env._env.spec._env_name == 'CartPole':
            min_data = tf.constant([-4.8, -4, -0.41, -4], dtype=tf_type)
            max_data = tf.constant([+4.8, +4, +0.41, +4], dtype=tf_type)
        elif env._env.spec._env_name == 'MountainCar':
            min_data = tf.constant([-1.2, -0.07], dtype=tf_type)
            max_data = tf.constant([0.6, 0.07], dtype=tf_type)
        elif env._env.spec._env_name == 'Acrobot':
            #  self.min_data = np.array([-np.pi, -np.pi, -13.0, -22]).reshape(1, -1)
            #self.max_data = np.array([+np.pi, +np.pi, +13.0, +22]).reshape(1, -1)p
            min_data = tf.constant([-np.pi, -np.pi, -13.0, -22], dtype=tf_type)
            max_data = tf.constant([+np.pi, +np.pi, +13.0, +22], dtype=tf_type)

#         with tf.device("GPU:0"):

        state = env.reset()
        self.ftr_dim = (fourier_order + 1) ** len(state)
        fourier_kernel_mat = tf.reshape(
                tf.meshgrid(*([np.arange(fourier_order + 1)] * len(state))),
                shape=[-1, self.ftr_dim]
            )
        fourier_kernel_mat = tf.transpose(fourier_kernel_mat)
        fourier_kernel_mat = tf.cast(fourier_kernel_mat, dtype=tf_type)
        self.range_data = min_data - max_data
        self.min_data = min_data
        self.max_data = max_data
        self.fourier_kernel_mat = fourier_kernel_mat

    @tf.function
    def transform(self, s):
        s = (s - self.min_data) / self.range_data
        return tf.math.cos(np.pi * tf.linalg.matvec(self.fourier_kernel_mat, s))

class EnvWrapper:
    def __init__(self, env):
        self._env = env
        if env.spec._env_name == 'CartPole':
            self.min_clip, self.max_clip = [0, 500]
        elif env.spec._env_name == 'MountainCar':
            self.min_clip, self.max_clip = [-200, 0]
        elif env.spec._env_name == 'Acrobot':
            self.min_clip, self.max_clip = [-500, 0]

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        if self._env.spec._env_name == 'Acrobot':
            state = self._env.state
        return state, reward, done, info

    def reset(self):
        state = self._env.reset()
        if self._env.spec._env_name == 'Acrobot':
            state = self._env.state
        return state



def record_video(agent, fourier_basis, env_name):
    env = gym.wrappers.Monitor(gym.make(env_name), './recording/' + env_name, force=True)
    s = env.reset()
    s = fourier_basis.transform(tf.constant(s, dtype=tf.dtypes.double))
    while True:
        s,_,done,_ = env.step(agent.take_action(s).numpy())
        s = fourier_basis.transform(tf.constant(s, dtype=tf.dtypes.double))
        if done:
            break
    env.close()
