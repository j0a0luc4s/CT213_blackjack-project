import numpy as np

def reward_engineering_blackjack(observation, action, reward, next_observation, terminated):
    reward = 21 * reward
    if terminated and next_observation[0] <= 21:
        reward = reward + (next_observation[0] ** 2 / 21)
    return reward
