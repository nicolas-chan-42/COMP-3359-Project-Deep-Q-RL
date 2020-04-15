from gym.envs.registration import register
from .envs.connect_four_env_new import ConnectFourEnv, ResultType, Reward

register(
   	id='ConnectFour-v1',
   	entry_point='gym_connect_four.envs:ConnectFourEnv',
)