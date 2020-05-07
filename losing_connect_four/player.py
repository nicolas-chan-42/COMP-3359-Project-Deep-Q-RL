import random
from abc import ABC, abstractmethod
from operator import itemgetter
from random import Random
from typing import Optional, Dict

import numpy as np
import tensorflow as tf

from gym_connect_four import ConnectFourEnv
from losing_connect_four import DeepQNetwork
from losing_connect_four.deep_q_model import DeepQModel, ReplayMemory


class Player(ABC):
    """Abstract class for player"""

    @abstractmethod
    def __init__(self, env: ConnectFourEnv, name='Player'):
        self.name = name
        self.env = env

    def __repr__(self):
        return self.name

    def get_epsilon(self, total_step):
        pass

    @abstractmethod
    def get_next_action(self, state: np.ndarray, *args) -> int:
        pass

    def learn(self, state, action, next_state, reward, done, **kwargs):
        pass

    def save_model(self, model_dir: str):
        return NotImplementedError

    def load_model(self, model_dir: str):
        return NotImplementedError

    def write_summary(self, print_fn=print):
        return NotImplementedError

    def eval_mode(self, enable: bool = False):
        pass

    def reset(self, seed: Optional = None):
        pass


class RandomPlayer(Player):
    def __init__(self, env: ConnectFourEnv, name: str = 'RandomPlayer',
                 seed=None):
        super().__init__(env, name)
        self._seed = seed
        self._random = Random(seed)

    def get_next_action(self, *args, **kwargs) -> int:
        available_moves = self.env.available_moves()

        if not available_moves:
            raise ValueError('Unable to determine a valid move')

        # Choose one move from list of available moves
        action = self._random.choice(list(available_moves))

        return action

    def reset(self, seed: Optional = None) -> None:
        # For reproducibility of the random
        if seed:
            random.seed(seed)
        else:
            random.seed(self._seed)


class DeepQPlayer(Player):
    def __init__(self, env: ConnectFourEnv, params: Dict,
                 dqn_template: DeepQNetwork, is_eval=False,
                 name: str = "DeepQPlayer"):
        super().__init__(env, name)

        self.params = params

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.model = DeepQModel(env, params, dqn_template=dqn_template)
        self.is_eval = is_eval

    def get_epsilon(self, total_step):
        """Used decaying epsilon greedy exploration policy."""
        if self.is_eval:
            return 0

        eps_start = self.params["EPS_START"]
        eps_end = self.params["EPS_END"]
        eps_decay_steps = self.params["EPS_DECAY_STEPS"]

        if total_step <= eps_decay_steps:
            # Linear-decaying epsilon.
            return eps_start - total_step * (
                    eps_start - eps_end) / eps_decay_steps
        else:
            # Decayed epsilon.
            return eps_end

    def strategically_get_action(self, state, available_moves, epsilon: float):
        """
        Apply Epsilon-Greedy strategy when making move.

        Exploration with probability = epsilon;
        Exploitation with probability = (1-epsilon).

        :param state: state of Connect-Four environment
        :param available_moves: moves that are available and valid
        :param epsilon: probability of exploration.
        :return: a random action (exploration),
            or a DQN-decided action (exploitation).
        """
        # With prob. epsilon, (Exploration):
        #   select random action.
        if random.random() <= epsilon:
            return random.choice(list(available_moves))

        # With prob. 1 - epsilon, (Exploitation):
        #   select action with max predicted Q-Values of current state.
        else:
            q_values = self.model.predict(state)[0]
            valid_moves = [(i, q_values[i]) for i in available_moves]
            act, _ = max(valid_moves, key=itemgetter(1))
            return act

    # noinspection PyMethodOverriding
    def get_next_action(self, state, *, epsilon: float) -> int:
        # Add batch dimension and channel dimension for prediction.
        state = tf.convert_to_tensor(state)
        state = tf.reshape(state, (1, *self.observation_space, 1))

        action = self.strategically_get_action(
            state, self.env.available_moves(), epsilon)

        if self.env.is_valid_action(action):
            return action
        else:
            raise ValueError("Action is not valid!")

    def learn(self, state, action, next_state, reward, done,
              **kwargs):  # Should return loss
        """
        Use experiment replay to update the weights of the network.
        """
        if self.is_eval:
            return

        self.model.memorize(state, action, next_state, reward, done)

        epochs = kwargs.get("epochs", 1)
        self.model.experience_replay(epochs=epochs)

        # Update weights of Target DQN every STEPS_PER_TARGET_UPDATE.
        if "n_step" in kwargs:
            if kwargs["n_step"] % self.params["N_STEPS_PER_TARGET_UPDATE"] == 0:
                self.update_target_dqn_weights()
        else:
            raise ValueError("Keyword argument 'n_step' is missing")

    def update_target_dqn_weights(self):
        self.model.update_target_dqn_weights()

    def save_model(self, model_path):
        """Save the trained model using model directory as prefix."""
        self.model.save_model(model_path)

    def load_model(self, model_path):
        """Load the trained model using model directory as prefix."""
        self.model.load_model(model_path)

    def write_summary(self, print_fn=print):
        """Write summary of deep-Q model."""
        self.model.write_summary(print_fn=print_fn)

    def eval_mode(self, enable=False):
        self.is_eval = enable


class PretrainRandomPlayer(RandomPlayer):
    """Random Player for the use of Pre-training."""

    def __init__(self, env: ConnectFourEnv, memory: ReplayMemory,
                 name: str = 'PretrainRandomPlayer', seed=None):
        super().__init__(env, name, seed=seed)

        self.memory = memory

    def learn(self, state, action, next_state, reward, done,
              **kwargs):
        """Learning by simply memorizing."""
        self.memory.push(state, action, next_state, reward, done)


class PretrainRandomPlayerOnlyReward(PretrainRandomPlayer):
    def learn(self, state, action, next_state, reward, done,
              **kwargs):
        """Learning by simply memorizing (steps with reward only)."""
        if abs(reward) > 0:
            self.memory.push(state, action, next_state, reward, done)


class HumanPlayer(Player):
    def __init__(self, env: ConnectFourEnv, name: str = "HumanPlayer"):
        super().__init__(env, name)

    def get_next_action(self, state: np.ndarray, **kwargs) -> int:
        self.env.render()
        while True:
            action = int(input("Select your next action in 1-7: ")) - 1
            if self.env.is_valid_action(action):
                return action
            else:
                print("Action is not valid, please try again: ")
                continue
