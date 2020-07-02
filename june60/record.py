import gym
from PIL import Image

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, 'tmp/recording', force=True)
    env.reset()
    while True:
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            break
    env.close()
