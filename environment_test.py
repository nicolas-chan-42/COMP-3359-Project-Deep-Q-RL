import gym
from gym_connect_four import ConnectFourEnv

env = gym.make('ConnectFour-v1')

action = 3

env.step(action, render=True)
