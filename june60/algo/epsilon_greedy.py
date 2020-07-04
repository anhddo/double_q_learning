import numpy.random as npr
class EpsilonGreedy(object):
    def __init__(self, max_epsilon, min_epsilon, total_step, fraction, random_func, action_func):
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.random_func = random_func
        self.action_func = action_func
        self.epsilon_decay = (min_epsilon / max_epsilon) ** (1. / (fraction * total_step))
        self.epsilon = max_epsilon

    def action(self, state, step):
        self.epsilon = max(self.max_epsilon * self.epsilon_decay ** step, self.min_epsilon)
        if self.epsilon > npr.uniform():
            action = self.random_func()
        else:
            action = self.action_func(state)
        return action, {'epsilon': self.epsilon}
