import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x. https://cs231n.github.io/linear-classify/#softmax"""
    c = np.max(x, axis=0)
    shifted_x = x - c
    return np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis=0)



def argmax(vals):
    """
    Takes in a list of vals and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in vals
    """
    ties = np.argwhere(vals == np.amax(vals)).flatten()
    return np.random.choice(ties)


def utility(alpha, beta, gamma, r):
    if (r < 0):
        return -gamma * (-r)**beta
    else:
        return r**alpha

def plot_optimal_action_percentage(axis, optimal_action_percentage, agent_name):
    axis.plot(optimal_action_percentage)
    # axis.set_ylim([0, 1])
    axis.set_xlabel("Steps")
    axis.set_ylabel("% Optimal Action")
    axis.set_title("% Optimal Action of " + agent_name + " Agent")

def plot_average_reward(axis, average_reward, agent_name):
    axis.plot(average_reward)
    axis.set_xlabel("Steps")
    axis.set_ylabel("Average reward")
    axis.set_title("Average Reward of " + agent_name + " Agent")
