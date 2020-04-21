from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow_addons.optimizers import AdamW


def deep_q_network(self) -> Sequential:
    """
    Create a deep-Q neural network.
    :return: Tensorflow Deep-Q neural network model.
    """
    obs_space_card = self.observation_space[0] * self.observation_space[1]

    model = Sequential()
    model.add(Flatten(input_shape=self.observation_space))
    model.add(Dense(obs_space_card * 2, activation="relu"))
    model.add(Dense(obs_space_card * 2, activation="relu"))
    model.add(Dense(obs_space_card * 2, activation="relu"))
    model.add(Dense(obs_space_card * 2, activation="relu"))
    model.add(Dense(self.action_space, activation="linear"))

    # Used Adam optimizer to allow for weight decay
    optimizer = AdamW(lr=self.params["LR"],
                      weight_decay=self.params["LAMBDA"])
    model.compile(loss="mse", optimizer=optimizer)
    return model
