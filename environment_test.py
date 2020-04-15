import gym
from gym-connect-four import ConnectFourEnvNew

env = gym.make('ConnectFour-v1')

action = 3

env.step(action, render=True)
