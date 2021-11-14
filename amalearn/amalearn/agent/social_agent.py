# from amalearn.agent import AgentBase
# from abc import abstractmethod
# from queue import Queue
# from amalearn.social import Message

# class SocialAgent(AgentBase):
#     def __init__(self, id: str, container, environment, queue_max_size=100):
#         super(SocialAgent, self).__init__(id, environment)
#         if container is None:
#             raise Exception('The container cannot be None.')
#         self.container = container
#         self.container.register_agent(self, environment.id)
#         self.inbox = Queue(queue_max_size)
#         self.observables = []

#     # DO NOT CHANGE THIS METHOD
#     def request_observation(self, agent_id, env_id):
#         pass
    
#     # DO NOT CHANGE THIS METHOD
#     def cancel_observation(self, agent_id, env_id):
#         pass

#     # DO NOT CHANGE THIS METHOD
#     def send_message(self, message: Message):
#         self.container.enqueue_message(message)

#     def reset(self):
#         super().reset()

from amalearn.agent import AgentBase
import numpy as np
from utilities import core
from abc import abstractmethod

class SocialAgent(AgentBase):
    def __init__(self, id: str, environment, constant_stepsize=False, stepsize=0, c=np.sqrt(2), num_of_targets=1, betas=np.zeros(2)):
        super(SocialAgent, self).__init__(id, environment)
        self.stepsize = stepsize
        self.constant_stepsize = constant_stepsize
        self.betas = betas
        self.num_of_targets = num_of_targets
        self.targets_selected_actions = np.zeros(num_of_targets)
        self.c = c

        self.q_values = np.zeros(environment.available_actions())
        self.agents_arm_count = np.zeros(environment.available_actions())
        self.targets_arm_count = np.zeros((num_of_targets, environment.available_actions()))

    def take_action(self) -> (object, float, bool, object):
        available_actions = self.environment.available_actions()

        # print("before count", self.arm_count)
        # print("before q", self.q_values)
        
        t = 1 if np.sum(self.arm_count) == 0 else np.sum(self.arm_count) # shape = (1, 1)
        # print(t, np.log(t))
        # print(self.q_values + np.sqrt(0.5 * np.log(t) / (self.arm_count + 1)))
        l = [np.maximum((self.targets_arm_count[i] - self.arm_count) / t, 0) for i in range(self.num_of_targets)]
        l.insert(0, np.ones(available_actions))
        deltas = np.array(l) # shape = (num_of_targets + 1, num_of_actions)
        x = self.betas * deltas.T #(1, num_of_targets + 1) * (num_of_actions, num_of_targets + 1) -> (num_of_actions, num_of_targets + 1)
        # print(f"targets counts {self.targets_arm_count}")
        # print(f"deltas {deltas}")
        # print(f"betas * deltas {x}")
        s = np.sum(x, axis=1) #(num_of_actions, 1)
        # print(f"sum {s}")
        ucb = (self.c * np.sqrt(np.log(t + 1) / (self.arm_count + 1e-7))) * s #(num_of_actions, 1)
        ucb = ucb.flatten()
        # print(f"ucb before {ucb}")
        mask = self.arm_count == 0
        ucb[mask] = 1e20
        # print(f"ucb after {ucb}")
        # print(f"q {self.q_values}")
        # print(self.q_values + ucb)
        current_action = core.argmax(self.q_values + ucb)
        # print("greedy")

        obs, r, d, i = self.environment.step(current_action)

        target_index = 0
        for action in self.targets_selected_actions:
            # print(target_index, action)
            self.targets_arm_count[int(target_index), int(action)] += 1 
            target_index += 1

        self.arm_count[current_action] += 1

        stepsize = self.stepsize if self.constant_stepsize else 1 / self.arm_count[current_action]
        q = self.q_values[current_action]
        # print(r, type(r), q, type(q))
        self.q_values[current_action] = q + stepsize * (r - q) 

        # print("after count", self.arm_count)
        # print("after q", self.q_values)

        # print(obs, r, d, i)
        # self.environment.render()
        return obs, r, d, i   

    def reset(self):
        super().reset()
        self.q_values = np.zeros(self.environment.available_actions())
        self.arm_count = np.zeros(self.environment.available_actions())
        self.targets_arm_count = np.zeros((self.num_of_targets, self.environment.available_actions()))



