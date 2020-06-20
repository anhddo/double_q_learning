import numpy.random as npr
class EpsilonGreedy(object):
    def __init__(self, max_epsilon, min_epsilon, step, fraction, random_func, action_func):
        self.min_epsilon = min_epsilon
        self.random_func = random_func
        self.action_func = action_func
        self.epsilon_decay = (min_epsilon / max_epsilon) ** (1. / (fraction * step))
        self.epsilon = max_epsilon

    def action(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        if self.epsilon > npr.uniform():
            action = self.random_func()
        else:
            action = self.action_func(state)
        return action, {'epsilon': self.epsilon}
