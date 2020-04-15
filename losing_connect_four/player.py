from abc import ABC
from random import random

import numpy as np

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


# TODO: Didn't include state. May take a look later
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
