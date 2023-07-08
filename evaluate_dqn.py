import gymnasium as gym
from hyperparameters import hyperparameters
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from dqn_agent import DQNAgent

n_episodes = 1000

env = gym.make('Blackjack-v1', sab=True, render_mode="human")
hyperparameters['action_space_dim'] = env.action_space.n
hyperparameters['observation_space_dim'] = len(env.observation_space)

agent = DQNAgent(**hyperparameters)

reward_history = []

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

print(sum(reward_history)/len(reward_history))
