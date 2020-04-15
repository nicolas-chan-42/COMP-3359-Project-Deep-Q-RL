import random
from abc import ABC

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
from tensorflow.optimizers import Adam

from gym_connect_four import ConnectFourEnv


class Player(ABC):
    """Abstract class for player"""

    def __init__(self, env: 'ConnectFourEnv', name='Player'):
        self.name = name
        self.env = env

    def get_next_action(self, state: np.ndarray) -> int:
        pass

    def reset(self, episode=0, side=1):
        pass


class RandomPlayer(Player):
    def __init__(self, env, name='RandomPlayer', seed=None):
        super().__init__(env, name)
        self._seed = seed

    def get_next_action(self, *args, **kwargs) -> int:
        available_moves = self.env.available_moves()

        if not available_moves:
            raise ValueError('Unable to determine a valid move')

        # Choose one move from list of available moves
        action = random.choice(list(available_moves))

        return action

    def reset(self, episode: int = 0, side: int = 1) -> None:
        # For reproducibility of the random
        random.seed(self._seed)


class DeepQPlayer(Player):
    def __init__(self, env: ConnectFourEnv, params, name='DeepQPlayer'):
        super().__init__(env, name)

        fc_layer_params = (100,)

        # TODO:Can make one DQN using checkpoint
        self.net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=fc_layer_params)

        self.optimizer = Adam(learning_rate=params["LR"])

        train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        self.policy = self.agent.policy

    # TODO: Seems there is discrepancies between tf env and py env. Should solve this later
    def get_next_action(self, state) -> int:
        time_step = available_moves = self.env.available_moves()

        if not available_moves:
            raise ValueError('Unable to determine a valid move')

        action = self.policy.action(time_step)

        if self.env.is_valid_action(action):
            return action
