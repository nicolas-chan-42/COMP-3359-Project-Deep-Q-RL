import gym
from gym_connect_four.gym_connect_four import ConnectFourEnvNew

env = gym.make('ConnectFour-v1')

action = 3

env.step(action, render=True)
