import numpy.random as npr


class EpsilonGreedy(object):
    def __init__(self, args, random_func, action_func):
        self.min_epsilon = args.min_epsilon
        self.max_epsilon = args.max_epsilon
        self.random_func = random_func
        self.action_func = action_func
        self.epsilon_decay = (self.min_epsilon / self.max_epsilon) ** (1. / args.final_exploration_step)
        self.epsilon = max_epsilon

    def action(self, state, step):
        self.epsilon = max(self.max_epsilon * self.epsilon_decay ** step, self.min_epsilon)
        if self.epsilon > npr.uniform():
            action = self.random_func()
        else:
            action = self.action_func(state)
        return action, {'epsilon': self.epsilon}

class EpsilonGreedyAgent(object):
    def __init__(self, agent, args):
        self.min_epsilon = args.min_epsilon
        self.max_epsilon = args.max_epsilon
        self.random_func = random_func
        self.epsilon_decay = (min_epsilon / max_epsilon) ** (1. / final_exploration_step)
        self.epsilon = max_epsilon
        self.agent = agent

    def take_action(self, state, step):
        self.epsilon = max(self.max_epsilon * self.epsilon_decay ** step, self.min_epsilon)
        if self.epsilon > npr.uniform():
            action = self.random_func()
        else:
            action = self.action_func(state)
        return action, {'epsilon': self.epsilon}
