from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

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
total_reward = 0
total_losses = 0
n_lose = 0
all_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)

# Setup players.
random_player = RandomPlayer(env)
# with tf.device('/CPU:0'):
dq_player = DeepQPlayer(env, PARAMS)

players = {1: dq_player, 2: random_player}
player_id = 1
trainee_id = 1

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
    next_state, reward, done, _ = env.step(action)

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

        # Take the latest action in the deque. In endgame, winner here.
        next_state, reward, done, _ = env.step(action_hist[-1])

        # Store the resulting state to history
        state_hist.append(next_state)

        # Change player here
        player_id = env.change_player()
        player = players[player_id]

        # Update DQN weights. In endgame, loser here.
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

    # Both player have learnt all steps at the end. In endgame, winner here.
    reward *= -1
    player.learn(state_hist[-2], action_hist, state_hist[-1], reward, done,
                 n_step=total_step)

    # Adjust reward for trainee.
    # If winner is opponent, we give opposite reward to trainee.
    if player_id != trainee_id:
        reward *= -1  # adjust reward.

    n_lose += max(0, int(reward))  # zero if draw

    # TODO: have a list of 2 reward slots for the two players
    episode_reward += reward

    # Log the episode reward to aggregator
    total_reward += reward
    all_rewards[episode] = episode_reward
    cumulative_rewards[episode] = total_reward
    cumulative_losses[episode] = n_lose
    if (episode+1) % 100 == 0:
        print(f"Episode: {episode}")
        print(f"Cumulative Rewards: {total_reward}")
        print(f"Total Losses: {n_lose}")
        print(f"Total Steps: {total_step}")
        print("==========================")

print(f"Cumulative rewards in the end {all_rewards.sum()}")
print(f"Mean rewards in the end {all_rewards.sum()}")
print(f"Number of losses: {n_lose}")

# Visualize the training results
# Plot cumulative rewards
plt.plot(cumulative_rewards)
plt.show()

# Plot cumulative number of losses
plt.plot(cumulative_losses)
plt.show()
