from amalearn import environment
import numpy as np
from amalearn.agent import AgentBase
from utilities import core


class EpsilonGreedyAgent(AgentBase):
    def __init__(self, id, environment, epsilon, constant_stepsize=False, stepsize=0, q_initial_values=None, utility=lambda a, b, gamma, r: r, alpha=1, beta=1, gamma=1):
        super(EpsilonGreedyAgent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.constant_stepsize = constant_stepsize
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
            current_action = core.argmax(self.q_values)
            # print("greedy")

        obs, r, d, i = self.environment.step(current_action)
        u = self.utility(self.alpha, self.beta, self.gamma, r)
        # print(r, u)
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
        
