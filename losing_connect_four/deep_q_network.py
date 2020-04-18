""" Deep Q Network """
import random
from collections import deque
from operator import itemgetter

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow_addons.optimizers import AdamW

# Tensorflow GPU allocation.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class ReplayMemory:
    """
    A cyclic buffer to store transitions.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """Saves a transition."""
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetwork:
    """Deep-Q Neural Network model"""

    def __init__(self, env, params):
        self.params = params
        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.memory = ReplayMemory(params["REPLAY_BUFFER_MAX_LENGTH"])
        self.policy_dqn = self._deep_q_network()
        self.target_dqn = self._deep_q_network()
        assert (self.policy_dqn is not self.target_dqn)
        self.update_target_dqn_weights()

    def _deep_q_network(self):
        """
        Create a deep-Q neural network.
        :return: Tensorflow Deep-Q neural network model.
        """
        obs_space_card = self.observation_space[0] * self.observation_space[1]

        model = Sequential()
        model.add(Flatten(input_shape=self.observation_space))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))

        # Used Adam optimizer to allow for weight decay
        optimizer = AdamW(lr=self.params["LR"],
                          weight_decay=self.params["LAMBDA"])
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def update_target_dqn_weights(self):
        """Copy DQN weights from Policy DQN to Target DQN."""
        self.target_dqn.set_weights(self.policy_dqn.get_weights())

    def strategically_get_action(self, state, available_moves, epsilon):
        """
        Apply Epsilon-Greedy strategy when making move.

        Exploration with probability = epsilon;
        Exploitation with probability = (1-epsilon).

        :param state: state of Connect-Four environment
        :param available_moves: moves that are available and valid
        :param epsilon: probability of exploration.
        :return: a random action (exploration),
            or a DQN-decided action (exploitation).
        """
        # With prob. epsilon, (Exploration):
        #   select random action.
        if random.random() <= epsilon:
            return random.choice(list(available_moves))

        # With prob. 1 - epsilon, (Exploitation):
        #   select action with max predicted Q-Values of current state.
        # TODO: Copied from source code, need to refactor
        else:
            q_values = self.policy_dqn.predict(state)[0]
            valid_moves = [(i, q_values[i]) for i in available_moves]
            act = max(valid_moves, key=itemgetter(1))
            return act[0]

    def memorize(self, state, action, next_state, reward, done):
        """Push transition to memory."""
        self.memory.push(state, action, next_state, reward, done)

    def experience_replay(self):
        """ Update Q-value here """

        # Hyper-parameters used in this project
        gamma = self.params["GAMMA"]
        batch_size = self.params["BATCH_SIZE"]

        # Check if the length of the memory is enough
        if len(self.memory) < batch_size:
            return

        # Obtain a random sample from the memory
        batch = self.memory.sample(batch_size)

        # Update Q value (COPIED, slightly modified)
        for state, action, next_state, reward, done in batch:
            q_update = reward
            if not done:
                q_update += (gamma * np.amax(
                    self.target_dqn.predict(next_state)[0]))

            q_values = self.policy_dqn.predict(state)
            q_values[0][action] = q_update
            self.policy_dqn.fit(state, q_values, verbose=0)

        # TODO: Need to compute loss
        # loss = self.policy_dqn.loss()
