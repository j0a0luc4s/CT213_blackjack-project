import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

from dqn_agent import DQNAgent
from hyperparameters import hyperparameters
from tqdm import tqdm
from utils import plot

# evaluation parameters
n_episodes = 100
alpha = 0.02

# environment and agent's hyperparameters
env = gym.make('Blackjack-v1', sab=True, render_mode="human")
hyperparameters['action_space_dim'] = env.action_space.n
hyperparameters['observation_space_dim'] = len(env.observation_space)

agent = DQNAgent(**hyperparameters)

# model reading
if os.path.exists('blackjack.h5'):
    print('Loading weights from previous learning session.')
    agent.load("blackjack.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)

# data collection
fig, ax = plt.subplots(ncols = 2)

reward_history = [0.0]
average_history = [0.0]

overflow_count = 0
overflow_amount = 0

# main loop
for episode in tqdm(range(1, n_episodes + 1)):
    # reset environment
    observation, info = env.reset()
    done = False

    while not done:
        # act
        action = agent.get_action_greedy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # update
        observation = next_observation
        env.render()

    # data collection
    if next_observation[0] > 21:
        overflow_count = overflow_count + 1
        overflow_amount = overflow_amount + next_observation[0] - 21

    reward_history.append(reward)
    average_history.append((1 - alpha)*average_history[-1] + alpha*reward)

    if episode % 10 == 0:
        plot(reward_history, ax[0], "Reward History")
        plot(average_history, ax[1], "Average History")
        fig.show()

print(f'Win Percentage: {reward_history.count(1)/len(reward_history)}')
print(f'Tie percentage: {reward_history.count(0)/len(reward_history)}')
print(f'Loss percentage: {reward_history.count(-1)/len(reward_history)}')
print(f'Overflow Count: {overflow_count}')
print(f'Overflow Amount: {overflow_amount}')
