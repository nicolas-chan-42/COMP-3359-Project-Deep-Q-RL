"""
Integrate DQN components to build DQN here.
DQN built here will then be used by deep_q_model as dqn_template.
"""

from .deep_q_network_components.abc import DeepQNetwork
from .deep_q_network_components.loss_functions import LossFuncMixinMse
from .deep_q_network_components.networks import (
    Simple512Net, SimpleDefaultNet, CnnShrinkNet, CnnNoShrinkNet,
    PlaceholderNet,
)
from .deep_q_network_components.optimizers import (
    OptimizerMixinAdam, OptimizerMixinRMSProp, OptimizerMixinSGD,
)


class PlaceholderSgdDqn(PlaceholderNet,
                        OptimizerMixinSGD, LossFuncMixinMse, DeepQNetwork):
    """
    Placeholder network for loading model use.

    Optimizer: Adam;
    Loss Function: MSE.
    """
    pass


class SimpleDefaultAdamDqn(SimpleDefaultNet,
                           OptimizerMixinAdam, LossFuncMixinMse, DeepQNetwork):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 4 Dense(2 * obs_space)
    - Dense(output: action_space)

    Optimizer: Adam;
    Loss Function: MSE.
    """
    pass


class Simple512RmsPropDqn(Simple512Net,
                          OptimizerMixinRMSProp, LossFuncMixinMse,
                          DeepQNetwork):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 4 Dense(512)
    - Dense(output: action_space)

    Optimizer: RMSProp;
    Loss Function: MSE.
    """
    pass


class Simple512SgdDqn(Simple512Net,
                      OptimizerMixinSGD, LossFuncMixinMse, DeepQNetwork):
    """
    Architecture:

    - Flatten(input: observation_space)
    - 4 Dense(512)
    - Dense(output: action_space)

    Optimizer: SGD;
    Loss Function: MSE.
    """
    pass


class CnnShrinkSgdDqn(CnnShrinkNet,
                      OptimizerMixinSGD, LossFuncMixinMse, DeepQNetwork):
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

    Optimizer: SGD;
    Loss Function: MSE.
    """
    pass


class CnnNoShrinkageSgdDqn(CnnNoShrinkNet,
                           OptimizerMixinSGD, LossFuncMixinMse, DeepQNetwork):
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

    Optimizer: SGD;
    Loss Function: MSE.
    """
    pass
