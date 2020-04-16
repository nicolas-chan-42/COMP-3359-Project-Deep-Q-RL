import gym

from losing_connect_four.player import RandomPlayer, DeepQPlayer

"""Hyper-parameters"""
PARAMS = {"ENV_NAME": "ConnectFour-v1",
          "LR": 0.001,
          "REPLAY_BUFFER_MAX_LENGTH": 100000,
          "BATCH_SIZE": 32,
          "EPS_START": 1,
          "EPS_END": 0.01,
          "EPS_DECAY_STEPS": 10000,
          "GAMMA": 0.95,
          "LAMBDA": 0.001,
          "N_EPISODES": 2000}

""" Main Training Loop """
env = gym.make(PARAMS["ENV_NAME"])
state = env.reset()

done = False

# For logging
n_step = 1000  # set to 0 later
all_rewards = []

# TODO: Need to finish changing players
# Setup players.
random_player = RandomPlayer(env)
dq_player = DeepQPlayer(env, PARAMS)

players = {1: random_player,
           2: dq_player}
player = players[1]

# Logging for the episode

episode_reward = 0

# TODO: Main training loop
# TODO: Save model
# Inside ONE episode:
while not done:
    action = player.get_next_action(state, n_step=n_step)
    next_state, reward, done, _ = env.step(action)

    # Update DQN weights
    player.learn(state, action, next_state, reward, done)

    # Update training result at the end for the next episode
    n_step += 1
    state = next_state
    episode_reward += reward

    env.render()
    player_id = env.change_player()
    player = players[player_id]

print(episode_reward)
