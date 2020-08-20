import numpy.random as npr


class EpsilonGreedy(object):
    def __init__(self, args, random_func=None, action_func=None):
        self.min_epsilon = args.min_epsilon
        self.max_epsilon = args.max_epsilon
        self.random_func = random_func
        self.action_func = action_func
        final_exploration_step = 1
        if hasattr(args, "final_exploration_step"):
            final_exploration_step = args.final_exploration_step
        self.epsilon_decay = (self.min_epsilon / self.max_epsilon) ** (1. / final_exploration_step)
        self.epsilon = self.max_epsilon

    def get_epsilon(self, step):
        epsilon = max(self.max_epsilon * self.epsilon_decay ** step, self.min_epsilon)
        return epsilon

    def action(self, state, step):
        epsilon = self.get_epsilon(step)
        if epsilon > npr.uniform():
            action = self.random_func()
        else:
            action = self.action_func(state)
        return action, {'epsilon': epsilon}
