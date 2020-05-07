"""Network structure component to be used to build Deep-Q Networks."""

from tensorflow.keras.layers import (
    Flatten, Dense, ZeroPadding2D, Conv2D,
    Activation, Dropout,
)
from tensorflow.keras.models import Sequential

from losing_connect_four.deep_q_network_components.abc import NetworkMixin


class PlaceholderNet(NetworkMixin):
    """Placeholder network for loading model."""

    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        net = Sequential()
        net.add(Flatten(input_shape=observation_space))
        net.add(Dense(action_space, activation="linear"))
        return net


class SimpleDefaultNet(NetworkMixin):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 4 Dense(2 * obs_space)
    - Dense(output: action_space)
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


class Simple512Net(NetworkMixin):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 4 Dense(512)
    - Dense(output: action_space)
    """

    def _create_network(self, observation_space, action_space, params,
                        *args, **kwargs):
        net = Sequential()
        net.add(Flatten(input_shape=observation_space))
        net.add(Dense(512, activation="relu"))
        net.add(Dense(512, activation="relu"))
        net.add(Dense(512, activation="relu"))
        net.add(Dense(512, activation="relu"))
        net.add(Dense(action_space, activation="linear"))
        return net


class CnnShrinkNet(NetworkMixin):
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
    - Dense(512, activation="relu"))
    - Dense(256, activation="relu"))
    - Dense(128, activation="relu"))
    - Dense(64, activation="relu"))
    - Dense(action_space, activation="softmax"))
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
        net.add(Dense(512, activation="relu"))  # -> 512
        net.add(Dense(256, activation="relu"))  # -> 256
        net.add(Dense(128, activation="relu"))  # -> 128
        net.add(Dense(64, activation="relu"))  # -> 64
        net.add(Dense(action_space, activation="softmax"))  # -> 7

        return net


class CnnNoShrinkNet(NetworkMixin):
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
    - Dense(512, activation="relu"))
    - Dense(512, activation="relu"))
    - Dense(512, activation="relu"))
    - Dense(512, activation="relu"))
    - Dense(action_space, activation="softmax"))
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
        net.add(Dense(512, activation="relu"))  # -> 512
        net.add(Dense(512, activation="relu"))  # -> 512
        net.add(Dense(512, activation="relu"))  # -> 512
        net.add(Dense(512, activation="relu"))  # -> 512
        net.add(Dense(action_space, activation="softmax"))  # -> 7

        return net
