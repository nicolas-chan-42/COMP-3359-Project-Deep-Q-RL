from abc import ABC
from enum import Enum, unique
from typing import Tuple, NamedTuple, Optional

import gym
import numpy as np
import pygame
from gym import error, spaces

from gym_connect_four.envs.render import render_board


@unique
class ResultType(Enum):
    """
    Result type after each step in the environment.
    """
    NONE = None
    DRAW = 0
    WIN = 1

    def __eq__(self, other):
        """
        Need to implement this due to an unfixed bug in Python
        since 2017: https://bugs.python.org/issue30545
        """
        return self.value == other.value


class Reward:
    """
    A data class storing designed reward level.
    """
    NONE = 0
    DRAW = 0
    LOSS = 1
    WIN = -1


# noinspection PyShadowingNames
class ConnectFourEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, window_width=512, window_height=512):
        super().__init__()

        self.board_shape = (6, 7)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=self.board_shape,
                                            dtype=int)
        self.action_space = spaces.Discrete(self.board_shape[1])
        self.reward = Reward()

        self.__n_step = 0
        self.__current_player = 1
        self.__board = np.zeros(self.board_shape, dtype=int)

        # for visualisation and rendering.
        self.__player_color = 1
        self.__screen = None
        self.__window_width = window_width
        self.__window_height = window_height
        self.__rendered_board = self._update_board_render()

    class StepResult(NamedTuple):
        """
        Stores the result type after the step of action.
        Initialised as a namedtuple with res_type as the only field.
        """
        res_type: ResultType

        def get_reward(self):
            if self.res_type is ResultType.NONE:
                return Reward.NONE
            elif self.res_type is ResultType.DRAW:
                return Reward.DRAW
            elif self.res_type is ResultType.WIN:
                return Reward.WIN

        def is_done(self):
            return self.res_type != ResultType.NONE

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Perform a step according to action.
        :param action: the id of action.
        :return: A tuple of board state, reward, whether it's done, and info.
        """
        step_result = self._step(action)
        reward = step_result.get_reward()
        done = step_result.is_done()
        info = {"n_step": self.__n_step}
        return self.board, reward, done, info

    def _step(self, action: int) -> StepResult:
        """
        Apply action to the board.
        :param action: the id of action.
        :return: The result type of the step (NONE/DRAW/WIN1/WIN2).
        """
        result = ResultType.NONE

        # Raise exception if the action is invalid.
        if not self.is_valid_action(action):
            raise Exception(
                'Unable to determine a valid move! '
                'Maybe invoke at the wrong time?'
            )

        # Check empty position on board and perform action by filling
        for index in list(reversed(range(self.board_shape[0]))):
            if self.__board[index][action] == 0:
                self.__board[index][action] = self.__current_player
                self.__n_step += 1
                break

        # Check if board is completely filled
        if np.count_nonzero(self.__board[0]) == self.board_shape[1]:
            result = ResultType.DRAW
        else:
            # Check win condition
            if self.is_win_state():
                result = ResultType.WIN

        return self.StepResult(result)

    @property
    def board(self):
        return self.__board.copy()

    def reset(self, board: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset board to initial state, optionally setting as required state.

        :param board: board state to be set to.
        :return: board state after reset.
        """
        self.__current_player = 1
        self.__n_step = 0
        if board is None:
            self.__board = np.zeros(self.board_shape, dtype=int)
        else:
            self.__board = board
        self.__rendered_board = self._update_board_render()
        return self.board

    def render(self, mode: str = 'console',
               close: Optional[bool] = False) -> None:
        """
        Render the Connect-Four board.

        :param mode: "console" if render to console;
            "human" if render to PyGame window.
        :param close: whether to quit PyGame window or not.
        :return: None
        """
        if mode == 'console':
            replacements = {
                self.__player_color: 'A',
                0: ' ',
                -1 * self.__player_color: 'B'
            }

            def render_line(line):
                line = "|".join([f"{replacements[x]:^3}" for x in line])
                line = f"|{line}|"
                return line

            hline = '|---+---+---+---+---+---+---|'
            print(f"{self.__n_step}. Player {self.current_player}")
            print(hline)
            for line in np.apply_along_axis(render_line, axis=1,
                                            arr=self.__board):
                print(line)
            print(hline)
            print()

        elif mode == 'human':
            if self.__screen is None:
                pygame.init()
                self.__screen = pygame.display.set_mode(
                    (round(self.__window_width), round(self.__window_height)))

            if close:
                pygame.quit()

            self.__rendered_board = self._update_board_render()
            frame = self.__rendered_board
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.rotate(surface, 90)
            self.__screen.blit(surface, (0, 0))

            pygame.display.update()
        else:
            raise error.UnsupportedMode()

    # noinspection PyMethodMayBeStatic
    def close(self) -> None:
        pygame.quit()

    @property
    def current_player(self) -> int:
        """ Current Player ID (Player 1 or 2) """
        if self.__current_player == 1:
            return 1
        else:
            return 2

    def change_player(self) -> int:
        """
        Change current player in the environment.
        :return: Player ID.
        """
        self.__current_player *= -1
        return self.current_player

    def is_valid_action(self, action: int) -> bool:
        return self.__board[0][action] == 0

    def _update_board_render(self) -> np.ndarray:
        return render_board(self.__board,
                            image_width=self.__window_width,
                            image_height=self.__window_height)

    def is_win_state(self) -> bool:
        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(self.__board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*self.__board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += self.__board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(self.__board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    def available_moves(self) -> frozenset:
        return frozenset(
            (i for i in range(self.board_shape[1]) if self.is_valid_action(i)))
