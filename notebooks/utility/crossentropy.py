import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np


class MyCrossentropy(Loss):

    def __init__(self, category_weights, beta=1):
        super(MyCrossentropy, self).__init__()
        self.category_weights = beta / np.mean(category_weights) * category_weights

    def call(self, y_true, y_pred):
        pixel_losses = self.category_weights * tf.where(y_true, tf.math.log(y_pred + 2**-16), 0) + \
                       tf.where(tf.math.logical_not(y_true), tf.math.log(1 - y_pred + 2**-16), 0)
        return - tf.math.reduce_mean(pixel_losses, axis=[1, 2, 3])
