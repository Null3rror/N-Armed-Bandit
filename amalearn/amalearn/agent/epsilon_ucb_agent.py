from amalearn import environment
import numpy as np
from amalearn.agent import AgentBase
from utilities import core


class EpsilonUCBAgent(AgentBase):
    def __init__(self, id, environment, epsilon, constant_stepsize=False, stepsize=0, c=np.sqrt(2), q_initial_values=None, utility=lambda a, b, gamma, r: r, alpha=1, beta=1, gamma=1):
        super(EpsilonUCBAgent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.constant_stepsize = constant_stepsize
        self.c = c
        self.utility = utility
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        
        self.q_initial_values = q_initial_values
        self.q_values = np.array(self.q_initial_values) if self.q_initial_values != None else np.zeros(environment.available_actions())
        self.arm_count = np.zeros(environment.available_actions())
        

    def take_action(self) -> (object, float, bool, object):
        available_actions = self.environment.available_actions()

        # print("before count", self.arm_count)
        # print("before q", self.q_values)
        rand = np.random.random()
        if rand < self.epsilon:
            current_action = np.random.randint(0, available_actions)
            # print("random")
        else:
            t = 1 if np.sum(self.arm_count) == 0 else np.sum(self.arm_count) # shape = (1, 1)
            ucb = self.c * np.sqrt(np.log(t + 1) / (self.arm_count + 1e-7))
            mask = self.arm_count == 0
            ucb[mask] = 1e20
            # print(t, np.log(t))
            # print(self.q_values + np.sqrt(0.5 * np.log(t) / (self.arm_count + 1)))
            current_action = core.argmax(self.q_values + ucb)
            # print("greedy")

        obs, r, d, i = self.environment.step(current_action)

        u = self.utility(self.alpha, self.beta, self.gamma, r)
        self.arm_count[current_action] += 1
        stepsize = self.stepsize if self.constant_stepsize else 1 / self.arm_count[current_action]
        q = self.q_values[current_action]
        # print(r, type(r), q, type(q))
        self.q_values[current_action] = q + stepsize * (u - q) 

        # print("after count", self.arm_count)
        # print("after q", self.q_values)

        # print(obs, r, d, i)
        # self.environment.render()
        return obs, r, d, i

    def reset(self):
        super().reset()
        self.q_values = np.array(self.q_initial_values) if self.q_initial_values != None else np.zeros(self.environment.available_actions())
        self.arm_count = np.zeros(self.environment.available_actions())

