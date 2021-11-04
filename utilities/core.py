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