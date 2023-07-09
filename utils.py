import numpy as np
import matplotlib.pyplot as plt

def plot_history(agent, reward_history, moving_average_history, fig_name):
    plt.plot(moving_average_history, 'r', linewidth = 2)
    plt.plot(reward_history, 'b')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend(['Reward Moving Average', 'Reward'])
    plt.show(block=False)
    plt.pause(0.1)
    plt.savefig(f'{fig_name}.png')
    agent.save("blackjack.h5")

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
    if observation[0] <= 10:
        if action == 1:
            reward += 0.5
        else:
            reward -= 5

    if observation[0] == 10:
        if observation[2] == True and action == 1:
            reward += 1

    if observation[0] == 21 and action != 0:
        reward -= 5

    if observation[0] > 21 or next_observation[0] > 21:
        reward -= 5

    if 17 <= observation[0] <= 21 and action == 0:
        reward += 2 * (observation[0] - 17)/(21 - 17)

    return reward
