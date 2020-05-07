"""Optimizer component to be used to build Deep-Q Networks."""

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from losing_connect_four.dqn_components.abc import OptimizerMixin


# Optimizers.
class OptimizerMixinAdam(OptimizerMixin):
    def _create_optimizer(self, params, *args, **kwargs):
        learning_rate = params["LR"]
        return Adam(learning_rate=learning_rate, *args, **kwargs)


class OptimizerMixinRMSProp(OptimizerMixin):
    def _create_optimizer(self, params, *args, **kwargs):
        learning_rate = params["LR"]
        return RMSprop(learning_rate=learning_rate, *args, **kwargs)


class OptimizerMixinSGD(OptimizerMixin):
    def _create_optimizer(self, params, *args, **kwargs):
        learning_rate = params["LR"]
        return SGD(learning_rate=learning_rate, *args, **kwargs)
