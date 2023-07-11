import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

from dqn_agent import DQNAgent
from hyperparameters import hyperparameters
from tqdm import tqdm
from utils import plot, reward_engineering_blackjack

n_episodes = 2000
alpha = 0.02

env = gym.make("Blackjack-v1", sab=True)
hyperparameters['action_space_dim'] = env.action_space.n
hyperparameters['observation_space_dim'] = len(env.observation_space)

agent = DQNAgent(**hyperparameters)

fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()

reward_history = [0.0]
average_history = [0.0]

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

        observation = next_observation

    reward_history.append(reward)
    average_history.append((1 - alpha)*average_history[-1] + alpha*reward)

    if episode % 10 == 0:
        plot(reward_history, ax0, 'Reward History')
        plot(average_history, ax1, 'Average History')
        fig0.savefig("reward.eps")
        fig0.savefig("reward.png")
        fig1.savefig("average.eps")
        fig1.savefig("average.png")
        agent.save("blackjack.h5")
