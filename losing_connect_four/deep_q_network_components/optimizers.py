"""Optimizer component to be used to build Deep-Q Networks."""

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from losing_connect_four.deep_q_network_components.abc import OptimizerMixin


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
