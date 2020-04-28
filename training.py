import os
from collections import deque
from statistics import mean
from typing import NamedTuple, List

import gym
import matplotlib.pyplot as plt
import numpy as np

# Must be put before any tensorflow import statement.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from losing_connect_four.deep_q_networks import SimpleFCSgdDqn
from losing_connect_four.player import RandomPlayer, DeepQPlayer, Player
from losing_connect_four.training import train_one_episode, pretrain, \
    load_model_to_players

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
    "N_EPISODES": 10,
    "EPOCHS_PER_LEARNING": 2,
    "EPOCHS_PER_PRETRAIN_LEARNING": 2,
    "N_STEPS_PER_TARGET_UPDATE": 1000,
    "PRETRAIN": False,
    "PRETRAIN_UTILISATION_RATE": 0.95,
}

CONFIG = {
    # Please use "/" only for filepath and directory paths.
    # Use None as placeholder.
    "MODEL_DIR": "saved_models", # Input directory path here.
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
player1: Player = DeepQPlayer(env, PARAMS, SimpleFCSgdDqn)
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
# noinspection DuplicatedCode
total_step = 0

# Reward.
total_reward = 0
recent50_rewards = deque(maxlen=50)
recent200_rewards = deque(maxlen=200)

moving_avg50_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
moving_avg200_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)

# Losses.
total_losses = 0
recent50_losses = deque(maxlen=50)
recent200_losses = deque(maxlen=200)

moving_avg50_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
moving_avg200_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_avg_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)

"""Main training loop"""
# noinspection DuplicatedCode
print(f"Training through {PARAMS['N_EPISODES']} episodes")
print("-" * 30)

for episode in range(PARAMS["N_EPISODES"]):
    print(f"\rIn episode {episode + 1}", end="")

    # Train 1 episode.
    episode_reward, total_step = train_one_episode(env, players, PARAMS,
                                                   total_step)

    # Collect results from the one episode.
    episode_loss = int(episode_reward > 0)  # Count losses only.

    # Log the episode reward.
    total_reward += episode_reward
    recent50_rewards.append(episode_reward)
    recent200_rewards.append(episode_reward)
    # Log the cumulative and moving-average reward.
    moving_avg50_rewards[episode] = mean(recent50_rewards)
    moving_avg200_rewards[episode] = mean(recent200_rewards)
    cumulative_rewards[episode] = total_reward

    # Log the episode loss.
    total_losses += episode_loss
    recent50_losses.append(episode_loss)
    recent200_losses.append(episode_loss)
    # Log the cumulative and moving-average loss.
    moving_avg50_losses[episode] = mean(recent50_losses)
    moving_avg200_losses[episode] = mean(recent200_losses)
    cumulative_losses[episode] = total_losses
    cumulative_avg_losses[episode] = total_losses / (episode + 1)

    # Periodically print episode information.
    if (episode + 1) % CONFIG["N_EPISODE_PER_PRINT"] == 0:
        print(f"\rEpisode: {episode + 1}")
        print(f"Total Steps: {total_step}")
        print("-" * 25)
        # Reward.
        print(f"50-Episode Moving-Average Reward: "
              f"{moving_avg50_rewards[episode]}")
        print(f"200-Episode Moving-Average Reward: "
              f"{moving_avg200_rewards[episode]}")
        print(f"Average Reward: {total_reward / (episode + 1)}")
        print(f"Total Reward: {total_reward}")
        print("-" * 25)
        # Losses.
        print(f"50-Episode Moving-Average Losses: "
              f"{moving_avg50_losses[episode]}")
        print(f"200-Episode Moving-Average Losses: "
              f"{moving_avg200_losses[episode]}")
        print(f"Average Losses: {cumulative_avg_losses[episode]}")
        print(f"Total Losses: {total_losses}")
        print("=" * 25)

# Print training information.
print("\rIn the end of training,")
print(f"Total Steps: {total_step}")
print(f"Total Reward: {total_reward}")
print(f"Average Reward: {total_reward / PARAMS['N_EPISODES']}")
print(f"Total Number of Losses: {total_losses}")
print(f"Average Number of Losses: {total_losses / PARAMS['N_EPISODES']}")

"""Visualize the training results"""
PlotLine = NamedTuple("PlotLine", data_array=np.ndarray, label=str)
Figure = NamedTuple("Figure", title=str, lines=List[PlotLine])

plot_list: List[Figure] = [
    Figure("Cumulative Reward received over episodes",
           [PlotLine(cumulative_rewards, "Cum. Reward")]),

    Figure("Moving Average Reward received over episodes",
           [PlotLine(moving_avg50_rewards, "50-Episode M.A."),
            PlotLine(moving_avg200_rewards, "200-Episode M.A.")]),

    Figure("Cumulative Number of Losses over episodes",
           [PlotLine(cumulative_losses, "Cum. Losses")]),

    Figure("Losing Rate over episodes",
           [PlotLine(moving_avg50_losses, "50-Episode M.A."),
            PlotLine(moving_avg200_losses, "200-Episode M.A."),
            PlotLine(cumulative_avg_losses, "Full Average")]),
]

plt.rcParams["figure.facecolor"] = "white"
for figure in plot_list:
    plot_title = figure.title

    for plot_line in figure.lines:
        data_array = plot_line.data_array
        plt.plot(data_array, "-", label=plot_line.label)
        # Plot average line.
        plt.hlines(y=data_array.mean(), xmin=0, xmax=len(data_array) - 1,
                   colors=plt.gca().lines[-1].get_color(),
                   linestyles="dashed")
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.show()

"""Save Trained Models and Summary"""
save_model = CONFIG.get("SAVE_MODEL", None)
if save_model:
    # Save trained model
    player = players[players["trainee_id"]]
    player.save_model(f"{CONFIG['MODEL_DIR']}{save_model}")

    # Save model summary
    with open(f"{CONFIG['MODEL_DIR']}{save_model}.txt", "w") as file:
        file.write(f"{'Hyper-parameters'.center(65, '_')}\n")
        for key, value in PARAMS.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write(f"{'Model Summary'.center(65, '_')}\n")

        player.write_summary(print_fn=lambda s: file.write(f"{s}\n"))

"""Model Evaluation"""
player1.eval_mode(enable=True)
player2.reset(3407)
print(f"Starting evaluation...", end="")
total_step = 0

# Reward.
total_reward = 0
recent50_rewards = deque(maxlen=50)
recent200_rewards = deque(maxlen=200)

moving_avg50_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
moving_avg200_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)

# Losses.
total_losses = 0
recent50_losses = deque(maxlen=50)
recent200_losses = deque(maxlen=200)

moving_avg50_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
moving_avg200_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
cumulative_avg_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)

# with tf.device('/CPU:0'):
"""Main evaluation loop"""
print(f"Evaluating through {PARAMS['N_EPISODES']} episodes")
print("-" * 30)

for episode in range(PARAMS["N_EPISODES"]):
    print(f"\rIn evaluation episode {episode + 1}", end="")

    # Train 1 episode.
    episode_reward, total_step = train_one_episode(env, players, PARAMS,
                                                   total_step)

    # Collect results from the one episode.
    episode_loss = int(episode_reward > 0)  # Count losses only.

    # Log the episode reward.
    total_reward += episode_reward
    recent50_rewards.append(episode_reward)
    recent200_rewards.append(episode_reward)
    # Log the cumulative and moving-average reward.
    moving_avg50_rewards[episode] = mean(recent50_rewards)
    moving_avg200_rewards[episode] = mean(recent200_rewards)
    cumulative_rewards[episode] = total_reward

    # Log the episode loss.
    total_losses += episode_loss
    recent50_losses.append(episode_loss)
    recent200_losses.append(episode_loss)
    # Log the cumulative and moving-average loss.
    moving_avg50_losses[episode] = mean(recent50_losses)
    moving_avg200_losses[episode] = mean(recent200_losses)
    cumulative_losses[episode] = total_losses
    cumulative_avg_losses[episode] = total_losses / (episode + 1)

    # Periodically print episode information.
    if (episode + 1) % CONFIG["N_EPISODE_PER_PRINT"] == 0:
        print(f"\rEpisode: {episode + 1}")
        print(f"Total Steps: {total_step}")
        print("-" * 25)
        # Reward.
        print(f"50-Episode Moving-Average Reward: "
              f"{moving_avg50_rewards[episode]}")
        print(f"200-Episode Moving-Average Reward: "
              f"{moving_avg200_rewards[episode]}")
        print(f"Average Reward: {total_reward / (episode + 1)}")
        print(f"Total Reward: {total_reward}")
        print("-" * 25)
        # Losses.
        print(f"50-Episode Moving-Average Losses: "
              f"{moving_avg50_losses[episode]}")
        print(f"200-Episode Moving-Average Losses: "
              f"{moving_avg200_losses[episode]}")
        print(f"Average Losses: {cumulative_avg_losses[episode]}")
        print(f"Total Losses: {total_losses}")
        print("=" * 25)

# Print evaluation information.
print("\rIn the end of training,")
print(f"Total Steps: {total_step}")
print(f"Total Reward: {total_reward}")
print(f"Average Reward: {total_reward / PARAMS['N_EPISODES']}")
print(f"Total Number of Losses: {total_losses}")
print(f"Average Number of Losses: {total_losses / PARAMS['N_EPISODES']}")

"""Visualize the training results"""
PlotLine = NamedTuple("PlotLine", data_array=np.ndarray, label=str)
Figure = NamedTuple("Figure", title=str, lines=List[PlotLine])

plot_list: List[Figure] = [
    Figure("Cumulative Reward received over evaluation episodes",
           [PlotLine(cumulative_rewards, "Cum. Reward")]),

    Figure("Moving Average Reward received over evaluation episodes",
           [PlotLine(moving_avg50_rewards, "50-Episode M.A."),
            PlotLine(moving_avg200_rewards, "200-Episode M.A.")]),

    Figure("Cumulative Number of Losses over evaluation episodes",
           [PlotLine(cumulative_losses, "Cum. Losses")]),

    Figure("Losing Rate over evaluation episodes",
           [PlotLine(moving_avg50_losses, "50-Episode M.A."),
            PlotLine(moving_avg200_losses, "200-Episode M.A."),
            PlotLine(cumulative_avg_losses, "Full Average")]),
]

plt.rcParams["figure.facecolor"] = "white"
for figure in plot_list:
    plot_title = figure.title

    for plot_line in figure.lines:
        data_array = plot_line.data_array
        plt.plot(data_array, "-", label=plot_line.label)
        # Plot average line.
        plt.hlines(y=data_array.mean(), xmin=0, xmax=len(data_array) - 1,
                   colors=plt.gca().lines[-1].get_color(),
                   linestyles="dashed")
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.show()
