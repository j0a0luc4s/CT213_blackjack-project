import numpy as np
import matplotlib.pyplot as plt

def plot(history, ax, name):
    ax.plot(history, 'r')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title(name)
    plt.pause(0.1)

def reward_engineering_blackjack(observation, action, reward, next_observation, terminated):
    '''
    action:
    0: stick
    1: hit

    observation: 
    3-tuple containing: the players current sum, the value of 
    the dealers one showing card (1-10 where 1 is ace), and whether 
    the player holds a usable ace (0 or 1).

    next_observation:
    observation after next action is done.

    action:
    epsilon-greedy action choosen by the agent to be done.
    '''

    # possible actions
    stick = 0
    hit = 1

    # destructuring observations
    total, dealer_card, usable_ace = observation
    next_total, next_dealer_card, next_usable_ace = next_observation

    # amplification of gym's reward
    reward = 100 * reward

    # reward for final hand value
    if terminated:
        if next_total > 21:
            reward = reward - 10 * (next_total - 21)
        else:
            reward = reward + 50 * (next_total / 21) ** 4

    # threshold below of which the agent should hit
    hit_threshold = [17, 11, 13, 8, 8, 8, 16, 16, 16, 17]
    if action == stick:
        reward = reward - 5 * (hit_threshold[dealer_card - 1] - 1 - total)
    if action == hit:
        reward = reward + 5 * (hit_threshold[dealer_card - 1] - total)

    # incentive for hitting on ace
    if total == 1 and action == hit:
        reward = reward + 20

    return reward
