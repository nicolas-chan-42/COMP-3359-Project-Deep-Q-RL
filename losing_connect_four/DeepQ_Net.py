""" Deep Q Network """
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from operator import itemgetter

import random

# Class of DQN model
class DQN():

    def __init__(self, env, params):

        self.params = params

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.model = Sequential()
        obs_space_card = self.observation_space[0] * self.observation_space[1]
        self.model.add(Flatten(input_shape=self.observation_space))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=params["LR"]))

    def act(self, state, available_moves, epsilon):
        # With prob. epsilon,
        # (Exploration) select random action.
        if random.random() <= epsilon:
            return random.choice(list(available_moves))

        # With prob. 1 - epsilon,
        # (Exploitation) select action with max predicted Q-Values of current state.
        else:
            q_values = self.model.predict(state)[0]
            vs = [(i, q_values[i]) for i in available_moves]
            act = max(vs, key=itemgetter(1))
            return act[0]



