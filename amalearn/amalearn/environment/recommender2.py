import numpy as np
import gym
from amalearn.environment import EnvironmentBase

class RecommenderEnvironment2(EnvironmentBase):
    def __init__(self, rewards, episode_max_length, id, container=None):
        state_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(len(rewards.columns))

        super(RecommenderEnvironment2, self).__init__(action_space, state_space, id, container)
        self.arms_rewards = rewards
        self.episode_max_length = episode_max_length
        self.indexes = np.zeros(len(rewards.columns))
        self.state = {
            'length': 0,
            'last_action': None
        }

    def calculate_reward(self, action):
        reward = self.arms_rewards[str(action)].values[int(self.indexes[action])]
        print(reward)
        self.indexes[action] += 1
        return reward


    def terminated(self):
        return np.any(self.indexes >= len(self.arms_rewards))


    def observe(self):
        return {}

    def get_info(self, action):
        return {"action": action}

    def available_actions(self):
        return self.action_space.n

    def next_state(self, action):
        self.state['length'] += 1
        self.state['last_action'] = action

    def reset(self):
        self.state['length'] = 0
        self.state['last_action'] = None
        self.indexes = np.zeros(len(self.arms_rewards.columns))

    def render(self, mode='human'):
        print('{}:\taction={}'.format(self.state['length'], self.state['last_action']))

    def close(self):
        return
        