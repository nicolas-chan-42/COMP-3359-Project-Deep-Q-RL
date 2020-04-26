# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from datetime import date
#
# import gym
# import matplotlib.pyplot as plt
# import numpy as np
#
# from losing_connect_four.player import RandomPlayer, DeepQPlayer
# from losing_connect_four.training import train_one_episode
#
# import tensorflow as tf
#
# """Hyper-parameters"""
# PARAMS = {"ENV_NAME": "ConnectFour-v1",
#           "LR": 0.001,
#           "REPLAY_BUFFER_MAX_LENGTH": 1024,
#           "BATCH_SIZE": 16,
#           "EPS_START": 1,
#           "EPS_END": 0.01,
#           "EPS_DECAY_STEPS": 10000,
#           "GAMMA": 0.95,
#           "LAMBDA": 0.001,
#           "N_EPISODES": 2000,
#           "EPOCHS_PER_LEARNING": 2,
#           "N_STEPS_PER_TARGET_UPDATE": 1000,
#           "TRAINEE_MODEL_NAME": "DeepQPlayer",
#           "OPPONENT_MODEL_NAME": "DeepQPlayer",
#           "IS_LOAD_MODEL": False,
#           "IS_SAVE_MODEL": False}
#
# """ Main Training Loop """
# print("Making Connect Four gym environment...")
# env = gym.make(PARAMS["ENV_NAME"])
# done = False
#
# with tf.device('/CPU:0'):
#     # Setup players.
#     random_player = RandomPlayer(env, seed=3359)
#     dq_player = DeepQPlayer(env, PARAMS)
#     # Try to load the saved player if any
#     try:
#         if PARAMS["IS_LOAD_MODEL"]:
#             dq_player.load_model()
#             print("Saved model loaded")
#     except (IOError, ImportError):
#         pass
#     players = {1: dq_player, 2: random_player}
#     player_id = 1
#     trainee_id = 1
#
#     # For logging
#     total_step = 0
#     total_reward = 0
#     n_lose = 0
#     all_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
#     cumulative_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
#     cumulative_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
#     cumulative_mean_rewards = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
#     cumulative_mean_losses = np.zeros(PARAMS["N_EPISODES"], dtype=np.float32)
#
#     # Main training loop
#     print(f"Training through {PARAMS['N_EPISODES']} episodes")
#     print("-" * 30)
#     for episode in range(PARAMS["N_EPISODES"]):
#         print(f"\rIn episode {episode + 1}", end="")
#         # Reset reward
#         episode_reward = 0
#
#         reward, total_step = train_one_episode(env, players, total_step, PARAMS)
#
#         n_lose += max(0, int(reward))  # zero if draw
#
#         # TODO: have a list of 2 reward slots for the two players
#         episode_reward += reward
#
#         # Log the episode reward to aggregator
#         total_reward += reward
#         all_rewards[episode] = episode_reward
#         cumulative_rewards[episode] = total_reward
#         cumulative_losses[episode] = n_lose
#         cumulative_mean_rewards[episode] = total_reward / (episode + 1)
#         cumulative_mean_losses[episode] = n_lose / (episode + 1)
#         if (episode + 1) % 100 == 0:
#             print(f"\rEpisode: {episode + 1}")
#             print(f"Cumulative Rewards: {total_reward}")
#             print(f"Cumulative Mean Rewards: {total_reward / (episode + 1)}")
#             print(f"Total Losses: {n_lose}")
#             print(f"Cumulative Mean Losses: {n_lose / (episode + 1)}")
#             print(f"Total Steps: {total_step}")
#             print("==========================")
#
# print(f"\rCumulative rewards in the end {all_rewards.sum()}")
# print(f"Mean rewards in the end {all_rewards.mean()}")
# print(f"Number of losses: {n_lose}")
# print(f"Mean Losses: {n_lose / PARAMS['N_EPISODES']}")
#
# """Visualize the training results"""
# plot_list = (
#     # (all_rewards, "Reward received in each episode"),
#     (cumulative_rewards, "Cumulative Reward received over episodes"),
#     # (cumulative_mean_rewards,
#     #  "Averaged Cumulative Reward received over episodes"),
#     # (cumulative_losses, "Cumulative Number of Losses over episodes"),
#     (cumulative_mean_losses, "Losing Rate over episodes"),
# )
# plt.rcParams["figure.facecolor"] = "white"
# for data_sequence, plot_title in plot_list:
#     plt.plot(data_sequence, ".-")
#     # Plot average line.
#     plt.hlines(data_sequence.mean(), 0, len(data_sequence) - 1,
#                colors="g", linestyles="dashed")
#     plt.title(plot_title)
#     plt.grid()
#     plt.show()
#
# """Save Models and Summaries"""
# if PARAMS["IS_SAVE_MODEL"]:
#     # Save trained model
#     dq_player.save_model()
#
#     # Save model summary
#     with open(f"{date.today().strftime('%Y%m%d')}.txt", "w") as file:
#         file.write(f"{'Hyper-parameters'.center(65, '_')}\n")
#         for key, value in PARAMS.items():
#             file.write(f"{key}: {value}\n")
#         file.write("\n")
#         file.write(f"{'Model Summary'.center(65, '_')}\n")
#
#         dq_player.write_summary(print_fn=lambda s: file.write(f"{s}\n"))
