from collections import deque
from typing import Dict, Tuple

from gym_connect_four import ConnectFourEnv


def train_one_episode(env: ConnectFourEnv, players: Dict, params: Dict,
                      total_step: int) -> Tuple[float, int]:
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
    action = player.get_next_action(state, n_step=total_step)
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
        action_hist.append(player.get_next_action(state, n_step=total_step))

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
