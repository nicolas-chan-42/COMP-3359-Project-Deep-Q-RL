import random
from abc import ABC

import numpy as np

from gym_connect_four import ConnectFourEnv
from losing_connect_four.DeepQ_Net import DQN


class Player(ABC):
    """Abstract class for player"""

    def __init__(self, env: ConnectFourEnv, name='Player'):
        self.name = name
        self.env = env

    def get_next_action(self, state: np.ndarray, *args) -> int:
        pass

    def reset(self, episode=0, side=1):
        pass

    def learn(self, state, action, next_state, reward, done):
        pass


class RandomPlayer(Player):
    def __init__(self, env: ConnectFourEnv, name: str = 'RandomPlayer',
                 seed=None):
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
            # Linear-decaying epsilon.
            return eps_start - global_step * (
                    eps_start - eps_end) / eps_decay_steps
        else:
            # Decayed epsilon.
            return eps_end

    # TODO: Move epsilon to main training environment
    # noinspection PyMethodOverriding
    def get_next_action(self, state, *, n_step) -> int:
        epsilon = self.get_epsilon(n_step)
        state = np.reshape(state, [1] + list(self.observation_space))
        action = self.net.act(state, self.env.available_moves(), epsilon)
        if self.env.is_valid_action(action):
            return action

    def learn(self, state, action, next_state, reward, done) -> None:  # Should return loss
        """
        Use experiment replay to update the weights of the network
        """

        state = np.reshape(state, [1] + list(self.observation_space))
        next_state = np.reshape(next_state, [1] + list(self.observation_space))

        self.net.memorize(state, action, next_state, reward, done)

        self.net.experience_replay()
