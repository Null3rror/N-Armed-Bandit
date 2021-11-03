from amalearn.reward import RewardBase
import numpy as np

class GaussianUnifromReward(RewardBase):
    def __init__(self, mean, std, low, high, p1, p2):
        super(GaussianUnifromReward, self).__init__()
        self.mean = mean
        self.std = std
        self.low = low 
        self.high = high
        self.p1 = p1
        self.p2 = p2

        self.distbs = [
            lambda n: np.random.normal(loc=self.mean, scale=self.std),
            lambda n: np.random.uniform(self.low, self.high)
        ]

    def get_reward(self):
        distbIndex = np.random.choice(len(self.distbs), p=[self.p1, self.p2])
        return self.distbs[distbIndex]
