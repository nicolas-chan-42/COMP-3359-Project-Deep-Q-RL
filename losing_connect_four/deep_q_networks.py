from abc import ABC, abstractmethod

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential


class DeepQNetwork(ABC):
    """
    Network base-class.
    :return: Tensorflow Deep-Q neural network model.
    """

    @abstractmethod
    def create_network(self, observation_space, action_space, params,
                       *args, **kwargs):
        return NotImplementedError

    @abstractmethod
    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def create_optimizer(self, params):
        raise NotImplementedError

    @abstractmethod
    def create_loss_function(self):
        raise NotImplementedError


class SimpleDeepFCQNetwork(DeepQNetwork):
    def create_network(self, observation_space, action_space, params,
                       *args, **kwargs):
        model = self._create_network(
            observation_space, action_space, params, *args, **kwargs)

        # Used Adam optimizer to allow for weight decay
        loss_function = self.create_loss_function()
        optimizer = self.create_optimizer(params)

        model.compile(loss=loss_function, optimizer=optimizer)
        return model

    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        obs_space_card = observation_space[0] * observation_space[1]

        model = Sequential()
        model.add(Flatten(input_shape=observation_space))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(obs_space_card * 2, activation="relu"))
        model.add(Dense(action_space, activation="linear"))
        return model

    def create_loss_function(self):
        return mean_squared_error

    def create_optimizer(self, params):
        return Adam(lr=params["LR"], beta_2=params["LAMBDA"])


class SimplerFCDQN(SimpleDeepFCQNetwork):
    def create_optimizer(self, params):
        return RMSprop(lr=params["LR"])
