"""Loss Function component to be used to build Deep-Q Networks."""

from tensorflow.keras.losses import mean_squared_error

from losing_connect_four.dqn_components.abc import LossFunctionMixin


# Loss Functions.
class LossFuncMixinMse(LossFunctionMixin):
    def _create_loss_function(self, *args, **kwargs):
        return mean_squared_error
