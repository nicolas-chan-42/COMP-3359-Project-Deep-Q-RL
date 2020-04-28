from collections import deque
from math import log
from typing import Dict, Tuple

from gym_connect_four import ConnectFourEnv
from losing_connect_four.deep_q_model import ReplayMemory
from losing_connect_four.player import PretrainRandomPlayer, Player, DeepQPlayer


def train_one_episode(env: ConnectFourEnv, players: Dict, params: Dict,
                      total_step: int = 0) -> Tuple[float, int]:
    """
    Perform 1 training episode in ConnectFour Environment.

    :param env: Gym environment (ConnectFourEnv).
    :param players: A dictionary containing the instance of
        Player 1, Player 2, and the ID of player to be trained (trainee).
    :param params: Hyper-parameter dictionary.
    :param total_step: Total number of steps performed in env.
    :return: A tuple of final reward and updated total_step.
    """
    # noinspection PyRedeclaration
    state = env.reset()

    # Set player.
    player_id = 1
    player = players[player_id]
    trainee_id = players.get("trainee_id", 1)

    # Do one step ahead of the while loop
    # Initialize action history and perform first step.
    epsilon = player.get_epsilon(total_step=total_step)
    action = player.get_next_action(state, epsilon=epsilon)
    action_hist = deque([action], maxlen=2)
    next_state, reward, done, _ = env.step(action)

    # Initialize the state history and save the state and the next state
    state_hist = deque([state], maxlen=4)
    next_state *= -1  # Multiply -1 to change player perspective of game board.
    state_hist.append(next_state)
    state = next_state

    # Change player and enter while loop
    player_id = env.change_player()
    player = players[player_id]

    while not done:
        # Get current player's action.
        epsilon = player.get_epsilon(total_step=total_step)
        action_hist.append(player.get_next_action(state, epsilon=epsilon))

        # Take the latest action in the deque. In endgame, winner here.
        next_state, reward, done, _ = env.step(action_hist[-1])

        # Store the resulting state to history.
        # If the next player is player 2,
        #   Multiply next_state by -1 to change player perspective.
        if player_id == 1:
            next_state *= -1
        state_hist.append(next_state)

        # Change player here
        player_id = env.change_player()
        player = players[player_id]

        # Update DQN weights. In endgame, loser here.
        reward *= -1
        player.learn(state_hist[-3], action_hist[-2],  # state and action
                     state_hist[-1], reward, done,  # next state, reward, done
                     n_step=total_step, epochs=params["EPOCHS_PER_LEARNING"])

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
    player.learn(state_hist[-2], action_hist[-1],
                 state_hist[-1] * -1, reward, done,
                 # Multiply -1 to change owner of each move.
                 n_step=total_step, epochs=params["EPOCHS_PER_LEARNING"])

    # Adjust reward for trainee.
    # If winner is opponent, we give opposite reward to trainee.
    if player_id != trainee_id:
        reward *= -1  # adjust reward.

    total_step += 1

    return reward, total_step


def pretrain(env: ConnectFourEnv, params: Dict, player: DeepQPlayer):
    """Generate memory and perform experience play to player."""

    # If the player is not deepQ player, it's meaningless to pre-train.
    if not isinstance(player, DeepQPlayer):
        return

    print(f"Pre-train {player!r}")
    print("-"*30)
    memory_size = params["REPLAY_BUFFER_MAX_LENGTH"]
    replay_memory = ReplayMemory(memory_size)

    # Setup random players generating the memories.
    player1: Player = PretrainRandomPlayer(env, replay_memory, seed=3359)
    player2: Player = PretrainRandomPlayer(env, replay_memory, seed=4904)
    players = {1: player1, 2: player2}

    total_step = 0
    while total_step < memory_size:
        _, total_step = train_one_episode(env, players, params, total_step)
        print(f"\rPreparing Pre-train memory: {total_step + 1}", end="")
    print()

    # Pre-train starts here.
    # Pass in generated memories to player to be pre-trained.
    player.model.memory.memory = replay_memory.memory

    # Experience replay.
    utilisation_rate = 0.95
    batch_size = params["BATCH_SIZE"]
    # Solving 1 - utilisation_rate = (1 - batch_size / memory)**n for n.
    n_episode = int(
        log(1 - utilisation_rate) / log(1 - batch_size / memory_size))
    print(f"Pre-train {player!r} for {n_episode} episodes "
          f"to achieve {utilisation_rate:.0%} utilisation of prepared memory")

    epochs = params["EPOCHS_PER_PRETRAIN_LEARNING"]
    for episode in range(int(n_episode)):
        print(f"\rPre-training {player!r} for episode {episode + 1}", end="")
        player.model.experience_replay(epochs=epochs)
    print()

    print(f"Pre-trained {player!r}")
    print("="*30)
