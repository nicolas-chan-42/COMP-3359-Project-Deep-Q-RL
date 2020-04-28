""" Deep Q Network """
import random
from collections import deque
from typing import Dict, Union, List, Deque, NamedTuple, Optional, Type

import gym
import numpy as np
import tensorflow as tf

from losing_connect_four.deep_q_networks import DeepQNetwork

# Tensorflow GPU allocation.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"Number of physical_devices detected: {len(physical_devices)}")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Python type hinting.
State = Union[tf.Tensor, np.ndarray]


class MemoryItem(NamedTuple):
    state: State
    action: int
    next_state: State
    reward: float
    done: bool


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
        memory_item = MemoryItem(state, action, next_state, reward, done)
        self.memory.append(memory_item)

    def sample(self, batch_size: int) -> List:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DeepQModel:
    """Deep-Q Neural Network model"""

    def __init__(self, env: gym.Env, params: Dict,
                 dqn_template: Type[DeepQNetwork]):
        self.params = params
        self.observation_space: List[int] = env.observation_space.shape
        self.action_space: int = env.action_space.n

        self.memory = ReplayMemory(params["REPLAY_BUFFER_MAX_LENGTH"])

        self.dqn_template = dqn_template
        self.policy_dqn = dqn_template().create_network(
            self.observation_space, self.action_space, params)
        self.target_dqn = dqn_template().create_network(
            self.observation_space, self.action_space, params)

        self.update_target_dqn_weights()

    def update_target_dqn_weights(self):
        """Copy DQN weights from Policy DQN to Target DQN."""
        self.target_dqn.set_weights(self.policy_dqn.get_weights())

    def predict(self, state: State) -> List:
        return self.policy_dqn.predict(state)

    def memorize(self, state: State, action: int,
                 next_state: State, reward: float, done: bool):
        """Push transition to memory."""
        self.memory.push(state, action, next_state, reward, done)

    def experience_replay(self, epochs: Optional[int] = 1):
        """ Update Q-value here """

        # Hyper-parameters used in this project
        gamma = self.params["GAMMA"]
        batch_size = self.params["BATCH_SIZE"]

        # Check if the length of the memory is enough
        if len(self.memory) < batch_size:
            return

        # Obtain a random sample from the memory
        batch = self.memory.sample(batch_size)

        # Update Q value.
        # Stack up arrays as batches.
        batches = MemoryItem(*zip(*batch))
        state_batch = tf.stack(
            tf.convert_to_tensor(batches.state, dtype=tf.int32))
        action_batch = tf.stack(
            tf.convert_to_tensor(batches.action, dtype=tf.int32))
        next_state_batch = tf.stack(
            tf.convert_to_tensor(batches.next_state, dtype=tf.int32))
        reward_batch = tf.stack(
            tf.convert_to_tensor(batches.reward, dtype=tf.float32))
        done_batch = tf.stack(
            tf.convert_to_tensor(batches.done, dtype=tf.bool))

        # Add channel dimension.
        state_batch = tf.expand_dims(state_batch, axis=3)
        next_state_batch = tf.expand_dims(next_state_batch, axis=3)

        # Prepare flipped states and actions.
        # Flip state and action along y-axis of game board.
        state_batch_flip = tf.reverse(state_batch, axis=tf.constant([2]))
        action_batch_flip = 6 - action_batch
        next_state_batch_flip = tf.reverse(next_state_batch,
                                           axis=tf.constant([2]))

        # Concatenate non-flip with flip batches.
        state_batch_w_flip = tf.concat((state_batch, state_batch_flip), axis=0)
        action_batch_w_flip = tf.concat((action_batch, action_batch_flip),
                                        axis=0)
        next_state_batch_w_flip = tf.concat(
            (next_state_batch, next_state_batch_flip), axis=0)

        # Concatenate reward and done with itself to match batch size.
        reward_batch_w_flip = tf.concat((reward_batch, reward_batch), axis=0)
        done_batch_w_flip = tf.concat((done_batch, done_batch), axis=0)

        # Q_update = ((max_a' Q'(s',a') * gamma) if (not done) else 0) + reward
        bool_mask = tf.logical_not(done_batch_w_flip)
        q_update = tf.reduce_max(self.target_dqn.
                                 predict(next_state_batch_w_flip), axis=1)
        q_update *= gamma
        q_update = tf.where(condition=bool_mask,
                            x=q_update, y=tf.zeros_like(q_update))
        q_update += reward_batch_w_flip

        # Q(s,a) <- Q(s,a) + Q_update
        q_values = self.policy_dqn.predict(state_batch_w_flip)
        indices = tf.stack(
            [tf.range(batch_size * 2, dtype=tf.int32), action_batch_w_flip],
            axis=1)
        q_values = tf.tensor_scatter_nd_update(q_values, indices, q_update)

        self.policy_dqn.fit(state_batch_w_flip, q_values,
                            epochs=epochs, verbose=0)

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

        loss_function = self.dqn_template().create_loss_function()
        optimizer = self.dqn_template().create_optimizer(self.params)

        # Load policy and target DQN model and compile
        def load_model_architecture_and_weights(filename: str):
            with open(f"{filename}.json", 'r') as json_file:
                model = tf.keras.models.model_from_json(json_file.read())
            model.load_weights(f"{filename}.h5")
            model.compile(loss=loss_function, optimizer=optimizer)
            return model

        self.policy_dqn = load_model_architecture_and_weights(filename)
        self.target_dqn = load_model_architecture_and_weights(filename)

    def write_summary(self, print_fn=print):
        """
        Write the summary of model.
        :param print_fn: print function to use.
        """
        self.policy_dqn.summary(print_fn=print_fn)
