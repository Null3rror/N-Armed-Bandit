from amalearn.reward import RewardBase


class ConstantReward(RewardBase):
    def __init__(self, val):
        super(ConstantReward, self).__init__()
        self.value = val

    def get_reward(self):
        return self.value
