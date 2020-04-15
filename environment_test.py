import gym
import random
from abc import ABC

from gym_connect_four import ConnectFourEnv
import numpy as np

env = gym.make('ConnectFour-v1')

actions = np.zeros(6, dtype=int)
temps = []


class Player(ABC):
    '''Abstract class for player'''
    def __init__(self, env: 'ConnectFourEnv', name='Player'):
        self.name = name
        self.env = env

    def get_next_action(self, state: np.ndarray) -> int:
        pass

    def reset(self, episode = 0, side = 1):
        pass

class RandomPlayer():
    def __init__(self, env, name = 'RandomPlayer', seed= None):
        super().__init__(env, name)
        self._seed = seed

    def reset(self, episode: int = 0, side: int = 1) -> None:
        # For reproducibility of the random
        random.seed(self._seed)
        self._state = random.getstate()

for action in actions:
    temps.append(env.step(action))
    env.render()
    env.change_player()

for temp in temps:
    print(temp)
