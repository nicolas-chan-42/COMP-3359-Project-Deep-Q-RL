import gym

from losing_connect_four.player import RandomPlayer, DeepQPlayer

from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym

# Hyper-parameters
PARAMS = {"LR": 0.001,
          "REPLAY_BUFFER_MAX_LENGTH": 100000,
          "BATCH_SIZE" : 32,
          "EPS_START": 1,
          "EPS_END" : 0.01,
          "EPS_DECAY_STEPS": 10000}

env_name = 'ConnectFour-v1'
env = gym.make(env_name)
state = env.reset()
global_step = 1000
done = False

# TODO: Need to finish changing players
while not done:
    random_player = RandomPlayer(env)
    dq_player = DeepQPlayer(env, PARAMS)

    action = dq_player.get_next_action(state, global_step)

    state, reward, done, _ = env.step(action)

    env.render()
    env.change_player()
