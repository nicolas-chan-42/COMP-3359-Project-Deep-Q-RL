from abc import ABC, abstractmethod

from tensorflow.keras.layers import Flatten, Dense, Conv2D, Activation
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

    def create_network(self, observation_space, action_space, params,
                       *args, **kwargs):
        # Add channel dimension.
        observation_space = (*observation_space, 1)

        net = self._create_network(
            observation_space, action_space, params, *args, **kwargs)

        # Used Adam optimizer to allow for weight decay
        loss_function = self.create_loss_function()
        optimizer = self.create_optimizer(params)

        net.compile(loss=loss_function, optimizer=optimizer)
        return net

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


class OptimizerMixin(ABC):
    """"Optimizer ABC"""
    @abstractmethod
    def create_optimizer(self, params):
        raise NotImplementedError


class LossFunctionMixin(ABC):
    """"Loss Function ABC"""
    @abstractmethod
    def create_loss_function(self):
        raise NotImplementedError


# Optimizers.
class OptimizerMixinAdam(OptimizerMixin):
    def create_optimizer(self, params):
        return Adam(lr=params["LR"])


class OptimizerMixinRMSProp(OptimizerMixin):
    def create_optimizer(self, params):
        return RMSprop(lr=params["LR"])


class OptimizerMixinSGD(OptimizerMixin):
    def create_optimizer(self, params):
        return SGD(momentum=0.1, nesterov=True)


# Loss Functions.
class LossFuncMSEMixin(LossFunctionMixin):
    def create_loss_function(self):
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


class CnnDqn(LossFuncMSEMixin, OptimizerMixinAdam, DeepQNetwork):
    """
    Architecture:

    - Conv2D(16, kernel_size=4, strides=2, padding="valid",
             input_shape=observation_space)
    - Activation("relu")
    - Conv2D(32, kernel_size=3, strides=2, padding="same")
    - Activation("relu")
    - Conv2D(32, kernel_size=3, strides=2, padding="same")
    - Activation("relu")
    - Dropout(0.25)
    - Flatten()
    - Dense(action_space, activation="softmax")

    Optimizer: Adam;
    Loss Function: MSE.
    """
    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=4, strides=2, padding="valid",
                         input_shape=observation_space))
        model.add(Activation("relu"))
        # model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(Activation("relu"))
        # model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(action_space, activation="softmax"))

        return model
