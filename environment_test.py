from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np

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
          "N_EPISODES": 1000,
          "N_STEPS_PER_TARGET_UPDATE": 1000}

""" Main Training Loop """
env = gym.make(PARAMS["ENV_NAME"])
done = False

# For logging
total_step = 0
n_lose = 0
all_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.int16)

# Setup players.
random_player = RandomPlayer(env)
# with tf.device('/CPU:0'):
dq_player = DeepQPlayer(env, PARAMS)

players = {1: random_player, 2: dq_player}
player_id = 1
trainee_id = 2

# TODO: Save model
# Inside ONE episode:
for episode in range(PARAMS["N_EPISODES"]):
    # Reset reward
    episode_reward = 0

    # noinspection PyRedeclaration
    state = env.reset()

    # Player in position
    player = players[player_id]

    # Do one step ahead of the while loop
    # Log state and action histories

    # Initialize action history and perform first step
    action = player.get_next_action(state, n_step=total_step)
    action_hist = deque([action], maxlen=2)
    next_state, _, _, _ = env.step(action)

    # Initialize the state history and save the state and the next state
    state_hist = deque([state], maxlen=4)
    state_hist.append(next_state)
    state = next_state

    # Change player and enter while loop
    player_id = env.change_player()
    player = players[player_id]

    while not done:
        # Get current player's action.
        action_hist.append(player.get_next_action(state, n_step=total_step))

        # Take the latest action in the deque.
        next_state, reward, done, _ = env.step(action_hist[-1])

        # Store the resulting state to history
        state_hist.append(next_state)

        # Change player here
        player_id = env.change_player()
        player = players[player_id]

        # Update DQN weights.
        reward *= -1
        player.learn(state_hist[-3], action_hist[-2],  # state and action
                     state_hist[-1], reward, done,  # next state, reward, done
                     n_step=total_step)

        # Update training result at the end for the next step
        total_step += 1
        state = next_state

        # Render game board (NOT recommended with large N_EPISODES)
        # env.render()

    # Change player at the end of episode.
    player_id = env.change_player()
    player = players[player_id]

    # Both player have learnt all steps at the end.
    reward *= -1
    player.learn(state_hist[-2], action_hist, state_hist[-1], reward, done,
                 n_step=total_step)

    n_lose += max(0, int(reward))  # zero if draw

    # TODO: have a list of 2 reward slots for the two players
    episode_reward += reward
    all_rewards[episode] = episode_reward

print(all_rewards.mean(), all_rewards.sum())
print(f"Number of losses: {n_lose}")

# Visualize the training results
plt.plot(all_rewards)
