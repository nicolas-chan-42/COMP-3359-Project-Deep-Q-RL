import random
from abc import ABC
from operator import itemgetter

import numpy as np

from gym_connect_four import ConnectFourEnv
from losing_connect_four.deep_q_network import DeepQNetwork


class Player(ABC):
    """Abstract class for player"""

    def __init__(self, env: ConnectFourEnv, name='Player'):
        self.name = name
        self.env = env

    def get_next_action(self, state: np.ndarray, *args) -> int:
        pass

    def reset(self, episode=0, side=1):
        pass

    def learn(self, state, action, next_state, reward, done, **kwargs):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


class RandomPlayer(Player):
    def __init__(self, env: ConnectFourEnv, name: str = 'RandomPlayer',
                 seed=None):
        super().__init__(env, name)
        self._seed = seed

    def get_next_action(self, *args, **kwargs) -> int:
        available_moves = self.env.available_moves()

        if not available_moves:
            raise ValueError('Unable to determine a valid move')

        # Choose one move from list of available moves
        action = random.choice(list(available_moves))

        return action

    def reset(self, episode: int = 0, side: int = 1) -> None:
        # For reproducibility of the random
        random.seed(self._seed)


class DeepQPlayer(Player):
    def __init__(self, env: ConnectFourEnv, params, name='DeepQPlayer'):
        super().__init__(env, name)

        self.params = params

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.net = DeepQNetwork(env, params)

    def get_epsilon(self, global_step):
        """Used decaying epsilon greedy exploration policy"""

        eps_start = self.params["EPS_START"]
        eps_end = self.params["EPS_END"]
        eps_decay_steps = self.params["EPS_DECAY_STEPS"]

        if global_step <= eps_decay_steps:
            # Linear-decaying epsilon.
            return eps_start - global_step * (
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
        # TODO: Copied from source, need to refactor
        else:
            q_values = self.net.predict(state)[0]
            valid_moves = [(i, q_values[i]) for i in available_moves]
            act = max(valid_moves, key=itemgetter(1))
            return act[0]

    # TODO: Move epsilon to main training environment
    # noinspection PyMethodOverriding
    def get_next_action(self, state, *, n_step) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        epsilon = self.get_epsilon(n_step)

        action = self.strategically_get_action(
            state, self.env.available_moves(), epsilon)
        if self.env.is_valid_action(action):
            return action

    def learn(self, state, action, next_state, reward, done,
              **kwargs):  # Should return loss
        """
        Use experiment replay to update the weights of the network
        """

        state = np.reshape(state, [1] + list(self.observation_space))
        next_state = np.reshape(next_state, [1] + list(self.observation_space))

        self.net.memorize(state, action, next_state, reward, done)

        self.net.experience_replay()

        # Update weights of Target DQN every STEPS_PER_TARGET_UPDATE.
        if kwargs["n_step"] % self.params["N_STEPS_PER_TARGET_UPDATE"] == 0:
            self.update_target_dqn_weights()

    def update_target_dqn_weights(self):
        self.net.update_target_dqn_weights()

    def save_model(self):
        """Save the trained model using self.name as prefix"""
        self.net.save_model(self.name)

    def load_model(self):
        """Load the trained model using self.name as prefix"""
        self.net.load_model(self.name)
