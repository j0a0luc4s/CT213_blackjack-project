from collections import deque
import numpy as np
import random
from tensorflow.keras import models, layers, optimizers, activations, losses
#from keras import models, layers, optimizers, activations, losses

class DQNAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        action_space_dim: int, 
        observation_space_dim: int,
        buffer_size: int, 
        batch_size: int
    ):
        self.learning_rate = learning_rate

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.discount_factor = discount_factor

        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim

        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.model = self.make_model()

    def make_model(
        self,
    ) -> models.Sequential:
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape = (self.observation_space_dim, )))
        model.add(layers.Dense(24, activation=activations.relu))
        model.add(layers.Dense(24, activation=activations.relu))
        model.add(layers.Dense(self.action_space_dim, activation=activations.linear))
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate))

        return model

    def get_action(
        self, 
        observation: tuple,
        ) -> int:
        observation = np.reshape(observation, (1, self.observation_space_dim))
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_dim)
        return np.argmax(self.model.predict(observation))

    def get_action_greedy(
        self, 
        observation: tuple,
        ) -> int:
        observation = np.reshape(observation, (1, self.observation_space_dim))
        return np.argmax(self.model.predict(observation))

    def append_experience(
        self,
        observation: tuple,
        action: int,
        reward: float,
        next_observation: tuple,
        terminated: bool,
    ):
        observation = np.reshape(observation, (1, self.observation_space_dim))
        next_observation = np.reshape(next_observation, (1, self.observation_space_dim))
        self.replay_buffer.append((observation, action, reward, next_observation, terminated))

    def replay(
        self, 
    ):
        if len(self.replay_buffer) <= 2 * self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        observations, targets = [], []
        for observation, action, reward, next_observation, terminated in batch:
            target = self.model.predict(observation)
            if not terminated:
                target[0][action] = reward + self.discount_factor * np.max(self.model.predict(next_observation)[0])
            else:
                target[0][action] = reward
            observations.append(observation[0])
            targets.append(target[0])

        self.model.fit(np.array(observations), np.array(targets), epochs=1, verbose=0)
        self.decay_epsilon()

    def load(
        self, 
        name: str, 
    ):
        self.model.load_weights(name)

    def save(
        self, 
        name: str, 
    ):
        self.model.save_weights(name)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
