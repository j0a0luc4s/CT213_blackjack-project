import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from utils import plot

n_episodes = 100
alpha = 0.02

env = gym.make('Blackjack-v1', sab=True, render_mode="human")

fig, ax = plt.subplots(ncols = 2)

reward_history = [0.0]
average_history = [0.0]

overflow_count = 0
overflow_amount = 0

for episode in tqdm(range(1, n_episodes + 1)):
    observation, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        observation = next_observation
        env.render()

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
print(f'Tie Percentage: {reward_history.count(0)/len(reward_history)}')
print(f'Loss Percentage: {reward_history.count(-1)/len(reward_history)}')
print(f'Overflow Count: {overflow_count}')
print(f'Overflow Amount: {overflow_amount}')
