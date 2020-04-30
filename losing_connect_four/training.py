from collections import deque
from math import log
from pathlib import Path
from typing import Dict, Tuple, List, NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gym_connect_four import ConnectFourEnv
from losing_connect_four.deep_q_model import ReplayMemory
from losing_connect_four.player import PretrainRandomPlayer, Player, DeepQPlayer


def train_one_episode(env: ConnectFourEnv, params: Dict, players: Dict,
                      total_step: int = 0) -> Tuple[float, int]:
    """
    Perform 1 training episode in ConnectFour Environment.

    :param env: Gym environment (ConnectFourEnv).
    :param params: Hyper-parameter dictionary.
    :param players: A dictionary containing the instance of
        Player 1, Player 2, and the ID of player to be trained (trainee).
    :param total_step: Total number of steps performed in env.
    :return: A tuple of final reward and updated total_step.
    """
    # noinspection PyRedeclaration
    state = env.reset()

    # Set player.
    player_id = 1
    player = players[player_id]
    trainee_id = players.get("trainee_id", 1)

    # Extract PARAMS.
    epochs = params.get("EPOCHS_PER_LEARNING", 1)

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
                     n_step=total_step, epochs=epochs)

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
                 n_step=total_step, epochs=epochs)

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
    print("-" * 30)
    memory_size = params["REPLAY_BUFFER_MAX_LENGTH"]
    replay_memory = ReplayMemory(memory_size)

    # Setup random players generating the memories.
    player1: Player = PretrainRandomPlayer(env, replay_memory, seed=3359)
    player2: Player = PretrainRandomPlayer(env, replay_memory, seed=4904)
    players = {1: player1, 2: player2}

    total_step = 0
    while total_step < memory_size:
        _, total_step = train_one_episode(env, params, players, total_step)
        print(f"\rPreparing Pre-train memory: {total_step + 1}", end="")
    print()

    # Pre-train starts here.
    # Pass in generated memories to player to be pre-trained.
    player.model.memory.memory = replay_memory.memory

    # Prepare for experience replay.
    batch_size = params["BATCH_SIZE"]
    utilisation_rate = params["PRETRAIN_UTILISATION_RATE"]
    utilisation_rate = min(1, max(0, utilisation_rate))  # clip within [0,1].
    utilisation_rate = utilisation_rate if utilisation_rate < 1 else 0.9999

    # Solving 1 - utilisation_rate = (1 - batch_size / memory)**n for n.
    n_episode = int(
        log(1 - utilisation_rate) / log(1 - batch_size / memory_size))
    print(f"Pre-train {player!r} for {n_episode} episodes "
          f"to achieve {utilisation_rate:.0%} utilisation of prepared memory")

    # Experience replay.
    epochs = params["EPOCHS_PER_PRETRAIN_LEARNING"]
    for episode in range(int(n_episode)):
        print(f"\rPre-training {player!r} for episode {episode + 1}", end="")
        player.model.experience_replay(epochs=epochs)
    print()

    print(f"Pre-trained {player!r}")
    print("=" * 30)


def load_model_to_players(config: Dict, params: Dict, players: Dict):
    """Load trained model to player"""
    models_to_be_loaded = config.get("LOAD_MODEL", [None, None])

    for player_id, model_spec in enumerate(models_to_be_loaded, start=1):
        # If no model is requested to be loaded, continue to next.
        if not model_spec:
            continue

        player = players[player_id]

        # Pre-trained player is not considered for loading.
        if params.get("PRETRAIN") and player_id == players["trainee_id"]:
            print(f"{player!r} is pre-trained and thus not loaded")
            continue

        # Only DeepQPlayer can load model.
        if not isinstance(players[player_id], DeepQPlayer):
            print(f"Only Deep-Q Players can load model, "
                  f"but {player!r} is not and thus not loaded")
            continue

        # Preparations before loading model.
        if config.get("MODEL_DIR", None):
            directory_path = Path(config["MODEL_DIR"])
            directory_path.mkdir(parents=True, exist_ok=True)
        else:
            directory_path = Path(".")
        model_path = directory_path / model_spec

        if not model_path.with_suffix(".json").is_file():
            raise Exception(f"{model_path.with_suffix('.json')} is not found")
        if not model_path.with_suffix(".h5").is_file():
            raise Exception(f"{model_path.with_suffix('.h5')} is not found")

        # Try loading model
        try:
            player = players[player_id]
            player.load_model(f"{model_path}")
            print(f"Loaded model {model_spec} for {player!r}")
        except (IOError, ImportError) as err:
            print(f"Model {model_spec} CANNOT be loaded due to error")
            raise err


def save_model_from_player(config: Dict, params: Dict, players: Dict):
    """Save trained model of player."""
    save_model_spec = config.get("SAVE_MODEL", None)

    # If no model is requested to save, return.
    if not save_model_spec:
        return

    player = players[players["trainee_id"]]

    # Only DeepQPlayer can save model.
    if not isinstance(player, DeepQPlayer):
        print(f"Only Deep-Q Players can save model, "
              f"but {player!r} is not and thus not saved")
        return

    # Preparations before saving model.
    if config.get("MODEL_DIR", None):
        directory_path = Path(config["MODEL_DIR"])
        directory_path.mkdir(parents=True, exist_ok=True)
    else:
        directory_path = Path(".")
    model_path = directory_path / save_model_spec

    # Save trained model.
    player.save_model(f"{model_path}")

    # Save model summary
    with open(f"{model_path.with_suffix('.txt')}", "w") as file:
        file.write(f"{'Hyper-parameters'.center(65, '_')}\n")
        for key, value in params.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write(f"{'Model Summary'.center(65, '_')}\n")

        player.write_summary(print_fn=lambda s: file.write(f"{s}\n"))


class Record:
    """Class for storing episodic rewards and losses information."""

    def __init__(self, params: Dict, config: Dict, name: str, dtype=np.float32):
        self.name = name
        self.n_episode: int = params["N_EPISODES"]
        self.period: int = config["N_EPISODE_PER_PRINT"]

        # Compute the window periods for moving averages.
        averaging_windows: np.ndarray = np.asarray([
            self.n_episode,
            self.period,
            self.period * 2,
            self.period // 2,
            50, 100, 200, 500, 1000,
        ]).clip(min=1, max=self.n_episode)
        self.averaging_windows: np.ndarray = np.unique(averaging_windows)

        # Preallocate a Pandas series for storing records.
        self.records = pd.Series(
            data=np.zeros(shape=self.n_episode),
            dtype=dtype,
            name=name,
        )

    def add_record(self, episode: int, record: float):
        """Add record.
        :param episode: in which episode.
        :param record: the value of record.
        """
        self.records[episode] = record

    def moving_averages(self, period: int) -> pd.Series:
        """
        Get moving average series.
        :param period: the window period averaging for.
        :return: a Pandas series of moving average with
            specified window period.
        """
        return self.records.rolling(period, min_periods=1).mean()

    def cumulative_sums(self) -> pd.Series:
        """
        Get cumulative sum series.
        :return: a Pandas series of cumulative sums.
        """
        return self.records.cumsum()

    def recent_average(self, episode: int, period: int) -> float:
        """
        Get the average of the given period before and including a
        specified episode.

        :param episode: which episode to use.
        :param period: the length of period to compute mean.
        :return: mean of records in the given period before and including
            given episode.
        """
        upper_bound = episode
        lower_bound = max(episode - period + 1, 0)
        return self.records[lower_bound:upper_bound].mean()

    def print_info(self, episode: int):
        """
        Print moving averages, average, and total.
        :param episode: which episode is at.
        """
        name = self.name
        total = self.records[:episode].sum()

        # Print Moving Averages.
        for window in self.averaging_windows:
            print(f"{window}-Episode Moving-Average {name}: "
                  f"{self.recent_average(episode=episode, period=window)}")
        # Print Average.
        print(f"Average {name}: {total / (episode + 1)}")
        # Print Total.
        print(f"Total {name}: {total}")


# NameTuples for Plotting.
PlotLine = NamedTuple("PlotLine", data_array=np.ndarray, label=str)
Figure = NamedTuple("Figure", title=str, lines=List[PlotLine])


def create_plot_list(records: Sequence[Record]) -> List[Figure]:
    """
    Create a plot list for a sequence of Record for latter plotting.

    :param records: a sequence of Record to be plotted.
    :return: a list of Figure for the sequence of Record
        for latter plotting.
    """
    plot_list = []
    for sub_records in records:
        for figure in create_plot_sublist(sub_records):
            plot_list.append(figure)
    return plot_list


def create_plot_sublist(records: Record) -> List[Figure]:
    """
    Create a plot list of individual Record for latter plotting.
    This function cooperates with function create_create_plot_list.

    :param records: an individual Record to be plotted
    :return: a list of Figure for this individual Record
        for latter plotting.
    """
    name = records.name
    plot_sublist = [
        Figure(f"Cumulative {name} over episodes",
               [PlotLine(records.cumulative_sums().values,
                         "Cum. Losses")]),

        Figure(f"Average {name} over episodes",
               [PlotLine(records.moving_averages(period=window).values,
                         f"{window}-Episode M.A.")
                for window in records.averaging_windows])]
    return plot_sublist


def plot_records(plot_list: List[Figure]):
    """
    Plot records/data and corresponding average lines for each figure task in
    the plot list.

    :param plot_list: A list of records/data to be plotted.
    """
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
