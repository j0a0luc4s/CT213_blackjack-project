from dqn_agent import DQNAgent
import gymnasium as gym
from hyperparameters import hyperparameters
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import reward_engineering_blackjack

n_episodes = 300

env = gym.make("Blackjack-v1", sab=True)
hyperparameters['action_space_dim'] = env.action_space.n
hyperparameters['observation_space_dim'] = len(env.observation_space)

return_history = []

agent = DQNAgent(**hyperparameters)

for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    done = False

    cumulative_reward = 0.0
    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.append_experience(observation, action, reward, next_observation, terminated)
        print("before: {}".format(reward))
        reward = reward_engineering_blackjack(observation, action, reward, next_observation, terminated)
        cumulative_reward = agent.discount_factor*cumulative_reward + reward
        print("after: {}".format(reward))
        agent.replay()

        done = terminated or truncated
        observation = next_observation

    return_history.append(cumulative_reward)

    agent.decay_epsilon()



agent.save("blackjack.h5")
