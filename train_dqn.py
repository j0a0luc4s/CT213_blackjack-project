import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

from dqn_agent import DQNAgent
from hyperparameters import hyperparameters
from tqdm import tqdm
from utils import plot_history, reward_engineering_blackjack

n_episodes = 1000

env = gym.make("Blackjack-v1", sab=True)
hyperparameters['action_space_dim'] = env.action_space.n
hyperparameters['observation_space_dim'] = len(env.observation_space)

agent = DQNAgent(**hyperparameters)

# Checking if weights from previous learning session exists
if os.path.exists('blackjack.h5'):
    print('Loading weights from previous learning session.')
    agent.load("blackjack.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)

moving_average = 0.0
moving_average_history = []
reward_history = []

for episode in tqdm(range(1, n_episodes + 1)):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        reward = reward_engineering_blackjack(observation, action, reward, next_observation, terminated)
        reward_history.append(reward)

        agent.append_experience(observation, action, reward, next_observation, terminated)
        agent.replay()

        done = terminated or truncated
        alpha = 0.1
        moving_average = (1 - alpha) * moving_average + alpha * reward
        moving_average_history.append(moving_average)
        observation = next_observation

    if episode % 10 == 0:
        plot_history(agent, reward_history, moving_average_history, 'train_history')

    agent.decay_epsilon()

