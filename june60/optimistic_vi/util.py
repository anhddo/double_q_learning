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
        if env.spec._env_name == 'CartPole':
            min_data = tf.constant([-4.8, -4, -0.41, -4], dtype=tf_type)
            max_data = tf.constant([+4.8, +4, +0.41, +4], dtype=tf_type)
            min_clip, max_clip = [0, 500]
        elif env.spec._env_name == 'MountainCar':
            min_data = tf.constant([-1.2, -0.07], dtype=tf_type)
            max_data = tf.constant([0.6, 0.07], dtype=tf_type)
            min_clip, max_clip = [-200, 0]
        elif env.spec._env_name == 'Acrobot':
            min_data = tf.constant([-1., -1., -1., -1., -13.0, -22], dtype=tf_type)
            max_data = tf.constant([+1., +1., +1., +1., +13.0, +22], dtype=tf_type)
            min_clip, max_clip = [-500, 0]

        s = env.reset()
#         with tf.device("GPU:0"):
        d = (fourier_order + 1) ** len(s)
        fourier_kernel_mat = tf.reshape(
                tf.meshgrid(*([np.arange(fourier_order + 1)] * len(s))),
                shape=[-1, d]
            )
        fourier_kernel_mat = tf.transpose(fourier_kernel_mat)
        fourier_kernel_mat = tf.cast(fourier_kernel_mat, dtype=tf_type)
        self.range_data = min_data - max_data
        self.min_data = min_data
        self.max_data = max_data
        self.fourier_kernel_mat = fourier_kernel_mat
        env.close()

    @tf.function
    def transform(self, s):
        s = (s - self.min_data) / self.range_data
        return tf.math.cos(np.pi * tf.linalg.matvec(self.fourier_kernel_mat, s))


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
