from tensorflow.keras.models import Model
import tensorflow as tf


class EnsembleNet(Model):

    def __init__(self, models, **kwargs):
        super().__init__(**kwargs)
        self.models = models

    def call(self, inputs, training=None, mask=None):
        outputs = [model(inputs, training=training, mask=mask) for model in self.models]
        return tf.math.reduce_mean(outputs, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "models": self.models
        })
        return config
