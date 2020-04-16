import random
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from gym_connect_four import ConnectFourEnv

from losing_connect_four.DeepQ_Net import DQN



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

        self.params = params

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.net = DQN(env, params)

        # used decaying epsilon greedy exploration policy
    def get_epsilon(self, global_step):

        eps_start = self.params["EPS_START"]
        eps_end = self.params["EPS_END"]
        eps_decay_steps = self.params["EPS_DECAY_STEPS"]

        if global_step <= eps_decay_steps:
            # When global_step <= eps_decay_steps, epsilon is decaying linearly.
            return eps_start - global_step * (eps_start - eps_end) / eps_decay_steps
        else:
            # Otherwise, epsilon stops decaying and stay at its minimum value eps_end
            return eps_end

    def get_next_action(self, state, global_step):
        epsilon = self.get_epsilon(global_step)
        state = np.reshape(state, [1] + list(self.observation_space))
        action = self.net.act(state, self.env.available_moves(), epsilon)
        if self.env.is_valid_action(action):
            return action


