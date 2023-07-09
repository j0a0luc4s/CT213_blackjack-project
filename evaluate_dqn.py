import gymnasium as gym
from hyperparameters import hyperparameters
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from dqn_agent import DQNAgent
from utils import plot_history

n_episodes = 100

env = gym.make('Blackjack-v1', sab=True, render_mode="human")
hyperparameters['action_space_dim'] = env.action_space.n
hyperparameters['observation_space_dim'] = len(env.observation_space)

agent = DQNAgent(**hyperparameters)

reward_history = []
moving_average = 0.0
moving_average_history = []

# Checking if weights from previous learning session exists
if os.path.exists('blackjack.h5'):
    print('Loading weights from previous learning session.')
    agent.load("blackjack.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)

for episode in tqdm(range(1, n_episodes + 1)):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        reward_history.append(reward)
        alpha = 0.1
        moving_average = (1 - alpha) * moving_average + alpha * reward
        moving_average_history.append(moving_average)
        observation = next_observation
        env.render()

    if episode % 10 == 0:
        plot_history(agent, reward_history, moving_average_history, 'evaluate_history')

print(f'Win rate: {reward_history.count(1)/(len(reward_history))}')
