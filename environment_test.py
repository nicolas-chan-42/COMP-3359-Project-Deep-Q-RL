import gym
from gym_connect_four import ConnectFourEnv
import numpy as np

env = gym.make('ConnectFour-v1')

actions = np.zeros(6, dtype=int)
temps = []

for action in actions:
    temps.append(env.step(action))
    env.render()
    env.change_player()

for temp in temps:
    print(temp)
