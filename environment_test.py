import gym
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
total_step = 1000  # set to 0 later
all_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.int16)

# Setup players.
random_player = RandomPlayer(env)
dq_player = DeepQPlayer(env, PARAMS)

players = {1: random_player, 2: dq_player}
player_id = 1
trainee_id = 2

# Logging for the episode

episode_reward = 0

# TODO: Save model
# Inside ONE episode:
for episode in range(PARAMS["N_EPISODES"]):
    # noinspection PyRedeclaration
    state = env.reset()

    # Final state and action for prediction after an episode is end
    final_prev_state = state
    final_state = state
    final_action = 0

    while not done:
        player = players[player_id]
        action = player.get_next_action(state, n_step=total_step)
        next_state, reward, done, _ = env.step(action)

        # Update DQN weights.
        player.learn(state, action, next_state, reward, done,
                     n_step=total_step)
        # Update training result at the end for the next episode
        total_step += 1
        # Log the final states and action if finished
        if done:
            final_prev_state = state
            final_state = next_state
            final_action = action

        state = next_state

        # if not done, add specific rewards to the player
        if not done:
            episode_reward += reward

        # env.render()
        player_id = env.change_player()

    if player_id == 1:
        reward = 1  # Raw implementation of losing reward
    else:
        reward = -1  # Raw implementation of win reward
    episode_reward += reward

    dq_player.learn(final_prev_state, final_action, final_state, reward, done,
                    n_step=total_step)

    all_rewards[episode] = episode_reward

print(all_rewards.mean(), all_rewards.sum())
