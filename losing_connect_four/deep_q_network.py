""" Deep Q Network """
import random
from collections import deque
from typing import Dict, Union, List, Tuple, Deque

import gym
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.models import model_from_json
from tensorflow_addons.optimizers import AdamW

# Tensorflow GPU allocation.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"Number of physical_devices detected: {len(physical_devices)}")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Python type hinting.
State = Union[tf.Tensor, np.ndarray]
MemoryItem = Tuple[State, int, State, float, bool]


class ReplayMemory:
    """
    A cyclic buffer to store transitions.
    """

    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: Deque[MemoryItem] = deque(maxlen=capacity)

    def push(self, state: State, action: int,
             next_state: State, reward: float, done: bool):
        """Saves a transition."""
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DeepQNetwork:
    """Deep-Q Neural Network model"""

    def __init__(self, env: gym.Env, params: Dict):
        self.params = params
        self.observation_space: List[int] = env.observation_space.shape
        self.action_space: int = env.action_space.n

        self.memory = ReplayMemory(params["REPLAY_BUFFER_MAX_LENGTH"])
        self.policy_dqn = self._deep_q_network()
        self.target_dqn = self._deep_q_network()
        self.update_target_dqn_weights()

    # TODO: Isolate _deep_q_network to enhance extensibility (OCP).
    def _deep_q_network(self) -> Sequential:
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

    def predict(self, state: State) -> List:
        return self.policy_dqn.predict(state)

    def memorize(self, state: State, action: int,
                 next_state: State, reward: float, done: bool):
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

    def save_model(self, filename: str):
        """
        Save trained model

        :param filename: Usually the name of the player.
        """
        # Save structure and weights into different files

        # Save policy DQN model weights
        self.policy_dqn.save_weights(f"{filename}.h5")

        # Save policy DQN structures
        model_json = self.policy_dqn.to_json()

        with open(f"{filename}.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()

    def load_model(self, filename: str):
        """
        Load trained model.

        :param filename: Usually the name of the player
        """
        optimizer = AdamW(lr=self.params["LR"],
                          weight_decay=self.params["LAMBDA"])

        # Load policy and target DQN model and compile
        def load_model_architecture_and_weights(filename: str):
            with open(f"{filename}.json", 'r') as json_file:
                model = model_from_json(json_file.read())
            model.load_weights(f"{filename}.h5")
            model.compile(loss="mse", optimizer=optimizer)
            return model

        self.policy_dqn = load_model_architecture_and_weights(filename)
        self.target_dqn = load_model_architecture_and_weights(filename)

# TODO: Need to compute loss
# loss = self.policy_dqn.loss()
