import gym

from losing_connect_four.player import RandomPlayer, DeepQPlayer

from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym

# Hyper-parameters
PARAMS = {"LR": 0.001,
          "REPLAY_BUFFER_MAX_LENGTH": 100000}

env_name = 'ConnectFour-v1'
env = gym.make(env_name)

tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

state = env.reset()
done = False
# TODO: Need to finish changing players
while not done:
    random_player = RandomPlayer(env)
    dq_player = DeepQPlayer(tf_env, PARAMS)

    action = dq_player.get_next_action(state)

    state, reward, done, _ = env.step(action)

    env.render()
    env.change_player()
