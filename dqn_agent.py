from collections import deque
import numpy as np
import random
from tensorflow.keras import models, layers, optimizers, activations, losses
#from keras import models, layers, optimizers, activations, losses

class DQNAgent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.
        :param initial_epsilon: initial epsilon used in epsilon-greedy policy.
        :type initial_epsilon: float.
        :param epsilon_decay: decay of epsilon per iteration.
        :type epsilon_decay: float.
        :param final_epsilon: final epsilon used in epsilon-greedy policy.
        :type final_epsilon: float.
        :param discount_factor: discount factor of the actio-value neural network.
        :type discount_factor: float.
        :param action_space_dim: number of actions.
        :type action_space_dim: int.
        :param observation_space_dim: number of dimensions of the feature vector of the observation.
        :type observation_space_dim: int.
        :param buffer_size: size of the experience replay buffer.
        :type buffer_size: int.
        :param batch_size: size of the experience replay batch.
        :type batch_size: int.
        """
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
        """
        Makes the action-value neural network model using Keras.

        :return: action-value neural network.
        :rtype: Keras' model.
        """
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
        """
        Chooses an action using an epsilon-greedy policy.

        :param observation: current observation.
        :type observation: NumPy array with dimension (observation_space_dim, ).
        :return: chosen action.
        :rtype: int.
        """
        observation = np.reshape(observation, (1, self.observation_space_dim))
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_dim)
        return np.argmax(self.model.predict(observation))

    def get_action_greedy(
        """
        Chooses an action using a greedy policy.

        :param observation: current observation.
        :type observation: NumPy array with dimension (observation_space_dim, ).
        :return: chosen action.
        :rtype: int.
        """
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
        """
        Appends a new experience to the replay buffer (and forget an old one if the buffer is full).

        :param observation: observation.
        :type observation: NumPy array with dimension (observation_space_dim, ).
        :param action: action.
        :type action: int.
        :param reward: reward.
        :type reward: float.
        :param next_observation: next observation.
        :type next_observation: NumPy array with dimension (observation_space_sim, ).
        :param terminated: if the simulation is over after this experience.
        :type terminated: bool.
        """
        observation = np.reshape(observation, (1, self.observation_space_dim))
        next_observation = np.reshape(next_observation, (1, self.observation_space_dim))
        self.replay_buffer.append((observation, action, reward, next_observation, terminated))

    def replay(
        self, 
    ):
        # check if the replay buffer has enough iterations, then samples a batch
        if len(self.replay_buffer) <= 2 * self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)

        # uses the bellman equation to generate data for model training
        observations, targets = [], []
        for observation, action, reward, next_observation, terminated in batch:
            target = self.model.predict(observation)
            if not terminated:
                target[0][action] = reward + self.discount_factor * np.max(self.model.predict(next_observation)[0])
            else:
                target[0][action] = reward
            observations.append(observation[0])
            targets.append(target[0])

        # fits the model, then decay epsilon for the next iteration
        self.model.fit(np.array(observations), np.array(targets), epochs=1, verbose=0)
        self.decay_epsilon()

    def load(
        self, 
        name: str, 
    ):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.load_weights(name)

    def save(
        self, 
        name: str, 
    ):
        """
        Saves the neural network's weights to disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.save_weights(name)

    def decay_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
