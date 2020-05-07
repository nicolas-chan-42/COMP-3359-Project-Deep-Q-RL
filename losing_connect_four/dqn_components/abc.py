"""
Abstract Basic Classes for Deep-Q Network, network structure component,
optimizer component, and loss function component.
"""

from abc import ABC, abstractmethod


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


# Network Mixin.
class NetworkMixin(ABC):
    """Deep-Q network ABC"""

    @abstractmethod
    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        raise NotImplementedError


# Optimizer Mixin.
class OptimizerMixin(ABC):
    """Optimizer ABC"""

    @abstractmethod
    def _create_optimizer(self, params, *args, **kwargs):
        raise NotImplementedError


# Loss Function Mixin.
class LossFunctionMixin(ABC):
    """"Loss Function ABC"""

    @abstractmethod
    def _create_loss_function(self, *args, **kwargs):
        raise NotImplementedError
