from amalearn import environment
import numpy as np
from amalearn.agent import AgentBase
from utilities import core


class EpsilonGreedyAgent(AgentBase):
    def __init__(self, id, environment, epsilon, stepsize, constant_stepsize=False, q_initial_values=None):
        super(EpsilonGreedyAgent, self).__init__(id, environment)
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.constant_stepsize = constant_stepsize
        self.q_values = np.array(q_initial_values) if q_initial_values != None else np.zeros(environment.available_actions())
        self.arm_count = np.zeros(environment.available_actions())
        

    def take_action(self) -> (object, float, bool, object):
        available_actions = self.environment.available_actions()

        print("before count", self.arm_count)
        print("before q", self.q_values)
        rand = np.random.random()
        if rand < self.epsilon:
            current_action = np.random.randint(0, available_actions)
            print("random")
        else:
            current_action = core.argmax(self.q_values)
            print("greedy")

        obs, r, d, i = self.environment.step(current_action)

        self.arm_count[current_action] += 1
        stepsize = self.stepsize if self.constant_stepsize else 1 / self.arm_count[current_action]
        q = self.q_values[current_action]
        # print(r, type(r), q, type(q))
        self.q_values[current_action] = q + stepsize * (r - q) 
        
        print("after count", self.arm_count)
        print("after q", self.q_values)

        # print(obs, r, d, i)
        self.environment.render()
        return obs, r, d, i
