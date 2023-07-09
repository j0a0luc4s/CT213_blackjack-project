import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from utils import plot_moving_average_history, plot_reward_history

n_episodes = 100

env = gym.make('Blackjack-v1', sab=True, render_mode="human")

reward_history = []

for episode in tqdm(range(1, n_episodes + 1)):
    observation, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        if done:
            reward_history.append(reward)
        observation = next_observation
        env.render()

    if episode % 10 == 0:
        plt.plot(reward_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Moving Average')
        plt.show(block=False)
        plt.pause(0.1)

print(f'Win rate: {reward_history.count(1)/(len(reward_history))}')
