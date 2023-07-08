import gymnasium as gym
from hyperparameters import hyperparameters
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from dqn_agent import DQNAgent

n_episodes = 30

fig_format = 'png'
# fig_format = 'eps'
# fig_format = 'svg'

env = gym.make('Blackjack-v1', sab=True, render_mode="human")
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
return_history = []

for episodes in tqdm(range(1, n_episodes)):
    observation, info = env.reset()
    cumulative_reward = 0.0
    done = False
    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        observation = next_observation
        cumulative_reward = agent.discount_factor * cumulative_reward + reward
        done = terminated or truncated
    return_history.append(cumulative_reward)

print('Mean return: ', np.mean(return_history))

plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation.' + fig_format, format=fig_format)
