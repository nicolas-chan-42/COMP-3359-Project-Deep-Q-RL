from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow_addons.optimizers import AdamW


class DeepQNetwork:
    """
    Create a deep-Q neural network.
    :return: Tensorflow Deep-Q neural network model.
    """
    def create_model(self, observation_space, action_space, params,
                     *args, **kwargs):
        raise NotImplementedError

    def _create_model(self, observation_space, action_space, params,
                      *args, **kwargs):
        raise NotImplementedError

    def create_optimizer(self, params):
        raise NotImplementedError


class SimpleDeepFCQNetwork(DeepQNetwork):
    def create_model(self, observation_space, action_space, params,
                     *args, **kwargs):
        model = self._create_model(
            observation_space, action_space, params, *args, **kwargs)

        # Used Adam optimizer to allow for weight decay
        optimizer = self.create_optimizer(params)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def _create_model(self, observation_space, action_space, params,
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

    def create_optimizer(self, params):
        return AdamW(lr=params["LR"], weight_decay=params["LAMBDA"])
