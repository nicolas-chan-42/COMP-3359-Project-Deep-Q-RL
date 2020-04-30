import os

import gym
import numpy as np

# Must be put before any tensorflow import statement.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from losing_connect_four.deep_q_networks import SimpleFCSgdDqn
from losing_connect_four.player import RandomPlayer, DeepQPlayer, Player
from losing_connect_four.training import (
    train_one_episode, pretrain, load_model_to_players,
    Record, plot_records, create_plot_list, save_model_from_player
)

# import tensorflow as tf

"""Hyper-parameters"""
PARAMS = {
    "ENV_NAME": "ConnectFour-v1",
    "LR": 0.001,
    "REPLAY_BUFFER_MAX_LENGTH": 100_000,
    "BATCH_SIZE": 32,
    "EPS_START": 1,
    "EPS_END": 0.01,
    "EPS_DECAY_STEPS": 10000,
    "GAMMA": 0.95,
    "N_EPISODES": 50,
    "EPOCHS_PER_LEARNING": 2,
    "EPOCHS_PER_PRETRAIN_LEARNING": 2,
    "N_STEPS_PER_TARGET_UPDATE": 1000,
    "PRETRAIN": False,
    "PRETRAIN_UTILISATION_RATE": 0.95,
}

CONFIG = {
    # Please use "/" only for filepath and directory paths.
    # Use None as placeholder.
    "MODEL_DIR": "saved_models",  # Input directory path here.
    "LOAD_MODEL": ["DQPlayer_seed_3407", None],  # Input filename here.
    "SAVE_MODEL": "DQPlayer_seed_3407",  # Input filename here
    "N_EPISODE_PER_PRINT": 100,
}

"""Set-up Environment"""
print("\rMaking Connect-Four Gym Environment...", end="")
env = gym.make(PARAMS["ENV_NAME"])
print("\rConnect-Four Gym Environment Made")

"""Setup Players"""
# with tf.device('/CPU:0'):
# Setup players.
player1: Player = DeepQPlayer(env, PARAMS,
                              SimpleFCSgdDqn(momentum=0))
player2: Player = RandomPlayer(env, seed=3407)
players = {1: player1, 2: player2,
           "trainee_id": 1}

"""Pre-train Player"""
if PARAMS["PRETRAIN"]:
    # noinspection PyTypeChecker
    pretrain(env, PARAMS, players[players["trainee_id"]])

"""Load the saved player if requested"""
load_model_to_players(CONFIG, PARAMS, players)

"""Logging"""
total_step = 0

# Reward.
total_reward = 0
reward_records = Record(PARAMS, CONFIG, name="Rewards", dtype=np.float32)

# Losses.
total_losses = 0
loss_records = Record(PARAMS, CONFIG, name="Losses", dtype=np.int32)

"""Main training loop"""
print(f"Training through {PARAMS['N_EPISODES']} episodes")
print("-" * 30)

for episode in range(PARAMS["N_EPISODES"]):
    print(f"\rIn episode {episode + 1}", end="")

    # Train 1 episode.
    episode_reward, total_step = train_one_episode(
        env, players, PARAMS, total_step)

    # Collect results from the one episode.
    episode_loss = int(episode_reward > 0)  # Count losses only.

    # Log the episode reward.
    reward_records.add_record(episode, record=episode_reward)
    total_reward += episode_reward

    # Log the episode loss.
    loss_records.add_record(episode, record=episode_loss)
    total_losses += episode_loss

    # Periodically print episode information.
    if (episode + 1) % CONFIG["N_EPISODE_PER_PRINT"] == 0:
        print(f"\rEpisode: {episode + 1}")
        print(f"Total Steps: {total_step}")
        print("-" * 25)
        # Reward.
        reward_records.print_info(episode)
        print("-" * 25)
        # Losses.
        loss_records.print_info(episode)
        print("=" * 25)

# Print training information.
print("\rIn the end of training,")
print(f"Total Steps: {total_step}")
print(f"Total Reward: {total_reward}")
print(f"Average Reward: {total_reward / PARAMS['N_EPISODES']}")
print(f"Total Number of Losses: {total_losses}")
print(f"Average Number of Losses: {total_losses / PARAMS['N_EPISODES']}")
print("=" * 30)

"""Visualize the training results"""
plot_list = create_plot_list([reward_records, loss_records])
plot_records(plot_list)

"""Save Trained Models and Summary"""
save_model_from_player(CONFIG, PARAMS, players)
