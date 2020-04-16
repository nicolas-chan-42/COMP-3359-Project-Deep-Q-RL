""" Deep Q Network """
import random
from collections import deque
from operator import itemgetter

import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class ReplayMemory(object):
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


class DQN:
    """Deep-Q Neural Network model"""

    def __init__(self, env, params):

        self.params = params

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.memory = ReplayMemory(params["REPLAY_BUFFER_MAX_LENGTH"])

        obs_space_card = self.observation_space[0] * self.observation_space[1]

        self.model = Sequential()
        self.model.add(Flatten(input_shape=self.observation_space))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))

        self.model.compile(loss="mse", optimizer=Adam(lr=params["LR"]))

    def act(self, state, available_moves, epsilon):
        # With prob. epsilon,
        # (Exploration) select random action.
        if random.random() <= epsilon:
            return random.choice(list(available_moves))

        # With prob. 1 - epsilon,
        # (Exploitation) select action with max predicted Q-Values of current state.
        # TODO: Copied from source code, need to refactor
        else:
            q_values = self.model.predict(state)[0]
            vs = [(i, q_values[i]) for i in available_moves]
            act = max(vs, key=itemgetter(1))
            return act[0]

    # Push transition to memory
    def memorize(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    # Q value updated here
    def experience_replay(self):

        # Hyperparameters used in this project
        gamma = self.params["GAMMA"]
        batch_size = self.params["BATCH_SIZE"]

        # Check if the length of the memory is enough
        if len(self.memory) < batch_size:
            return

        # Obtain a random sample from the memory
        batch = self.memory.sample(batch_size)

        # Update Q value (COPIED)
        for state, action, next_state, reward, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + gamma * np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        # TODO: Need to compute loss
        loss = self.model.loss()
