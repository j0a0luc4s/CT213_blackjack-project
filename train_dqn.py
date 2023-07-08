import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from dqn_agent import DQNAgent
from hyperparameters import hyperparameters
from tqdm import tqdm
from utils import reward_engineering_blackjack

n_episodes = 1000
alpha = 0.01

env = gym.make("Blackjack-v1", sab=True)
hyperparameters['action_space_dim'] = env.action_space.n
hyperparameters['observation_space_dim'] = len(env.observation_space)

agent = DQNAgent(**hyperparameters)

moving_average = 0.0
moving_average_history = []

for episode in tqdm(range(1, n_episodes + 1)):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        reward = reward_engineering_blackjack(observation, action, reward, next_observation, terminated)

        agent.append_experience(observation, action, reward, next_observation, terminated)
        agent.replay()

        done = terminated or truncated
        if done:
            moving_average = (1 - alpha) * moving_average + alpha * reward
            moving_average_history.append(moving_average)
        observation = next_observation

    if episode % 10 == 0:
        plt.plot(moving_average_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Moving Average')
        plt.show(block=False)
        plt.pause(0.1)
        agent.save("blackjack.h5")

    agent.decay_epsilon()

