import numpy as np
from amalearn.agent import AgentBase
from utilities import core


class EpsilonGradientAgent(AgentBase):
    def __init__(self, id, environment, epsilon, stepsize, constant_stepsize=False):
        super(EpsilonGradientAgent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.constant_stepsize = constant_stepsize
        self.h_values = np.zeros(environment.available_actions())
        self.Rbar = 0
        self.p_values = core.softmax(self.h_values)
        self.arm_count = np.zeros(environment.available_actions())

    def take_action(self) -> (object, float, bool, object):
        available_actions = self.environment.available_actions()

        print("before count", self.arm_count)
        print("before h", self.h_values)
        rand = np.random.random()
        if rand < self.epsilon:
            current_action = np.random.randint(0, available_actions)
            print("random")
        else:
            print("strategy")
            self.p_values = core.softmax(self.h_values)
            current_action = np.random.choice(available_actions, p=self.p_values)

        obs, r, d, i = self.environment.step(current_action)


        self.arm_count[current_action] += 1

        self.Rbar = self.Rbar + (1 / np.sum(self.arm_count)) * (r - self.Rbar)
        stepsize = self.stepsize if self.constant_stepsize else 1 / self.arm_count[current_action]

        mask = np.zeros(available_actions)
        mask[current_action] = 1
        self.h_values = mask * (self.h_values + stepsize * (r - self.Rbar) * (1 - self.p_values)) + \
                        (1 - mask) * ((self.h_values - stepsize * (r - self.Rbar) * self.p_values)) 

        print("p vals", self.p_values)
        print("after count", self.arm_count)
        print("after h", self.h_values)
        # print(obs, r, d, i)
        self.environment.render()
        return obs, r, d, i
