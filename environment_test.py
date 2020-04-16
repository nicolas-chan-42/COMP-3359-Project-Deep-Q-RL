import gym

from losing_connect_four.player import RandomPlayer, DeepQPlayer

"""Hyper-parameters"""
PARAMS = {"LR": 0.001,
          "REPLAY_BUFFER_MAX_LENGTH": 100000,
          "BATCH_SIZE" : 32,
          "EPS_START": 1,
          "EPS_END" : 0.01,
          "EPS_DECAY_STEPS": 10000}

""" Main Training Loop """


env_name = 'ConnectFour-v1'
env = gym.make(env_name)
state = env.reset()

done = False

# For logging

n_step = 1000 # set to 0 later
all_rewards = []

# TODO: Need to finish changing players
random_player = RandomPlayer(env)
dq_player = DeepQPlayer(env, PARAMS)

while not done:
    action = dq_player.get_next_action(state, n_step)

    state, reward, done, _ = env.step(action)

    n_step += 1

    env.render()
    env.change_player()
