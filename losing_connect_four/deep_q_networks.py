from abc import ABC, abstractmethod

from tensorflow.keras.layers import (
    Flatten, Dense, ZeroPadding2D, Conv2D,
    Activation, Dropout,
)
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


# Abstract Basic Classes.
class DeepQNetwork(ABC):
    """
    Network Abstract Base Class / Mixin Class.

    * create_network(...) returns the compiled network.
    * _create_network(...) returns the network architecture.
    * create_optimizer(...) returns the optimizer to be used.
    * create_create_loss_function(...) returns the loss function to be used.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def create_network(self, observation_space, action_space, params):
        args = self.args
        kwargs = self.kwargs

        # Add channel dimension.
        observation_space = (*observation_space, 1)

        net = self._create_network(
            observation_space, action_space, params, *args, **kwargs)

        # Used Adam optimizer to allow for weight decay
        loss_function = self.create_loss_function()
        optimizer = self.create_optimizer(params)

        net.compile(loss=loss_function, optimizer=optimizer)
        return net

    def create_optimizer(self, params):
        args = self.args
        kwargs = self.kwargs

        return self._create_optimizer(params, *args, **kwargs)

    def create_loss_function(self):
        args = self.args
        kwargs = self.kwargs

        return self._create_loss_function(*args, **kwargs)

    @abstractmethod
    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _create_optimizer(self, params, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _create_loss_function(self, *args, **kwargs):
        raise NotImplementedError


class OptimizerMixin(ABC):
    """"Optimizer ABC"""
    @abstractmethod
    def _create_optimizer(self, params, *args, **kwargs):
        raise NotImplementedError


class LossFunctionMixin(ABC):
    """"Loss Function ABC"""
    @abstractmethod
    def _create_loss_function(self, *args, **kwargs):
        raise NotImplementedError


# Optimizers.
class OptimizerMixinAdam(OptimizerMixin):
    def _create_optimizer(self, params, *args, **kwargs):
        return Adam(learning_rate=params["LR"], *args, **kwargs)


class OptimizerMixinRMSProp(OptimizerMixin):
    def _create_optimizer(self, params, *args, **kwargs):
        return RMSprop(learning_rate=params["LR"], *args, **kwargs)


class OptimizerMixinSGD(OptimizerMixin):
    def _create_optimizer(self, params, *args, **kwargs):
        return SGD(learning_rate=params["LR"], *args, **kwargs)


# Loss Functions.
class LossFuncMSEMixin(LossFunctionMixin):
    def _create_loss_function(self, *args, **kwargs):
        return mean_squared_error


# Networks.
class SimpleDeepFCQNetwork(OptimizerMixinAdam, LossFuncMSEMixin, DeepQNetwork):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 5 Dense(2 * obs_space)
    - Dense(output: action_space)

    Optimizer: Adam;
    Loss Function: MSE.
    """
    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        # Compute number of slots in Connect-Four.
        obs_space_card = observation_space[0] * observation_space[1]

        net = Sequential()
        net.add(Flatten(input_shape=observation_space))
        net.add(Dense(obs_space_card * 2, activation="relu"))
        net.add(Dense(obs_space_card * 2, activation="relu"))
        net.add(Dense(obs_space_card * 2, activation="relu"))
        net.add(Dense(obs_space_card * 2, activation="relu"))
        net.add(Dense(action_space, activation="linear"))
        return net


class SimpleFCRMSPropDQN(OptimizerMixinRMSProp, SimpleDeepFCQNetwork):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 5 Dense(2 * obs_space)
    - Dense(output: action_space)

    Optimizer: RMSProp;
    Loss Function: MSE.
    """
    pass


class SimpleFCSgdDqn(OptimizerMixinSGD, SimpleDeepFCQNetwork):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 5 Dense(2 * obs_space)
    - Dense(output: action_space)

    Optimizer: SGD;
    Loss Function: MSE.
    """
    pass


class CnnDqn(LossFuncMSEMixin, OptimizerMixinSGD, DeepQNetwork):
    """
    Architecture:

    - ZeroPadding2D(padding=((1, 0), (0, 0)), input_shape=observation_space))
    - Conv2D(16, kernel_size=2, strides=1, padding="valid"))
    - Activation("relu"))
    - Conv2D(32, kernel_size=2, strides=1, padding="valid"))
    - Activation("relu"))
    - Conv2D(64, kernel_size=2, strides=1, padding="valid"))
    - Activation("relu"))
    - Flatten())
    - Dropout(0.25))
    - Dense(256, activation="relu"))
    - Dense(128, activation="relu"))
    - Dense(64, activation="relu"))
    - Dense(32, activation="relu"))
    - Dense(action_space, activation="softmax"))

    Optimizer: SGD;
    Loss Function: MSE.
    """
    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        net = Sequential()
        # Zero-pad the top row: 6x7x1 -> 7x7x1.
        net.add(ZeroPadding2D(padding=((1, 0), (0, 0)),
                              input_shape=observation_space))
        net.add(Conv2D(64, kernel_size=2, strides=1, padding="valid"))
        net.add(Activation("relu"))  # -> 6x6x64
        net.add(Flatten())  # -> 2304
        net.add(Dropout(0.25))  # -> 1728
        net.add(Dense(256, activation="relu"))  # -> 512
        net.add(Dense(128, activation="relu"))  # -> 256
        net.add(Dense(64, activation="relu"))  # -> 128
        net.add(Dense(32, activation="relu"))  # -> 64
        net.add(Dense(action_space, activation="softmax"))  # -> 7

        return net
