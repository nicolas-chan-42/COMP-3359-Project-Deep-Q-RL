from gym.envs.registration import register

from gym_connect_four.envs import ConnectFourEnv, ResultType, Reward

register(
    id='ConnectFour-v1',
    entry_point='gym_connect_four.envs:ConnectFourEnv',
)
