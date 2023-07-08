import numpy as np

def reward_engineering_blackjack(observation, action, reward, next_observation, terminated):
    reward = reward - 0.1*np.abs(21 - observation[0])
    return reward
