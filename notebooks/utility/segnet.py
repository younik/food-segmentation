from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import BatchNormalization
#from MaxPoolingWithArgmax2D import MaxPoolingWithArgmax2D
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.layers import Layer

class SegNet(Model):

    def __init__(self, n_labels, kernel=3, pool_size=(2, 2), output_mode="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.n_labels = n_labels
        self.kernel = kernel
        self.pool_size = pool_size
        self.output_mode = output_mode
        self.conv_chan = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

        self.conv_1 = Convolution2D(self.conv_chan[0], (kernel, kernel), padding="same")
        self.batch_1 = BatchNormalization()
        self.activation_1 = Activation("relu")
        self.conv_2 = Convolution2D(self.conv_chan[1], (kernel, kernel), padding="same")
        self.batch_2 = BatchNormalization()
        self.activation_2 = Activation("relu")

        self.pool_1 = MaxPooling2D(pool_size, padding='same')
        #self.pool_1 = MaxPoolingWithArgmax2D()

        self.conv_3 = Convolution2D(self.conv_chan[2], (kernel, kernel), padding="same")
        self.batch_3 = BatchNormalization()
        self.activation_3 = Activation("relu")
        self.conv_4 = Convolution2D(self.conv_chan[3], (kernel, kernel), padding="same")
        self.batch_4 = BatchNormalization()
        self.activation_4 = Activation("relu")

        self.pool_2 = MaxPooling2D(pool_size, padding='same')
        #self.pool_2 = MaxPoolingWithArgmax2D()

        self.conv_5 = Convolution2D(self.conv_chan[4], (kernel, kernel), padding="same")
        self.batch_5 = BatchNormalization()
        self.activation_5 = Activation("relu")
        self.conv_6 = Convolution2D(self.conv_chan[5], (kernel, kernel), padding="same")
        self.batch_6 = BatchNormalization()
        self.activation_6 = Activation("relu")
        self.conv_7 = Convolution2D(self.conv_chan[6], (kernel, kernel), padding="same")
        self.batch_7 = BatchNormalization()
        self.activation_7 = Activation("relu")

        self.pool_3 = MaxPooling2D(pool_size, padding='same')
        #self.pool_3 = MaxPoolingWithArgmax2D()

        self.conv_8 = Convolution2D(self.conv_chan[7], (kernel, kernel), padding="same")
        self.batch_8 = BatchNormalization()
        self.activation_8 = Activation("relu")
        self.conv_9 = Convolution2D(self.conv_chan[8], (kernel, kernel), padding="same")
        self.batch_9 = BatchNormalization()
        self.activation_9 = Activation("relu")
        self.conv_10 = Convolution2D(self.conv_chan[9], (kernel, kernel), padding="same")
        self.batch_10 = BatchNormalization()
        self.activation_10 = Activation("relu")

        self.pool_4 = MaxPooling2D(pool_size, padding='same')
        #self.pool_4 = MaxPoolingWithArgmax2D()

        self.conv_11 = Convolution2D(self.conv_chan[10], (kernel, kernel), padding="same")
        self.batch_11 = BatchNormalization()
        self.activation_11 = Activation("relu")
        self.conv_12 = Convolution2D(self.conv_chan[11], (kernel, kernel), padding="same")
        self.batch_12 = BatchNormalization()
        self.activation_12 = Activation("relu")
        self.conv_13 = Convolution2D(self.conv_chan[12], (kernel, kernel), padding="same")
        self.batch_13 = BatchNormalization()
        self.activation_13 = Activation("relu")

        self.pool_5 = MaxPooling2D(pool_size, padding='same')
        #self.pool_5 = MaxPoolingWithArgmax2D()

        # decoder

        self.unpool_1 = UpSampling2D(pool_size)

        self.conv_14 = Convolution2D(self.conv_chan[11], (kernel, kernel), padding="same")
        self.batch_14 = BatchNormalization()
        self.activation_14 = Activation("relu")
        self.conv_15 = Convolution2D(self.conv_chan[10], (kernel, kernel), padding="same")
        self.batch_15 = BatchNormalization()
        self.activation_15 = Activation("relu")
        self.conv_16 = Convolution2D(self.conv_chan[9], (kernel, kernel), padding="same")
        self.batch_16 = BatchNormalization()
        self.activation_16 = Activation("relu")

        self.unpool_2 = UpSampling2D(pool_size)

        self.conv_17 = Convolution2D(self.conv_chan[8], (kernel, kernel), padding="same")
        self.batch_17 = BatchNormalization()
        self.activation_17 = Activation("relu")
        self.conv_18 = Convolution2D(self.conv_chan[7], (kernel, kernel), padding="same")
        self.batch_18 = BatchNormalization()
        self.activation_18 = Activation("relu")
        self.conv_19 = Convolution2D(self.conv_chan[6], (kernel, kernel), padding="same")
        self.batch_19 = BatchNormalization()
        self.activation_19 = Activation("relu")

        self.unpool_3 = UpSampling2D(pool_size)

        self.conv_20 = Convolution2D(self.conv_chan[5], (kernel, kernel), padding="same")
        self.batch_20 = BatchNormalization()
        self.activation_20 = Activation("relu")
        self.conv_21 = Convolution2D(self.conv_chan[4], (kernel, kernel), padding="same")
        self.batch_21 = BatchNormalization()
        self.activation_21 = Activation("relu")
        self.conv_22 = Convolution2D(self.conv_chan[3], (kernel, kernel), padding="same")
        self.batch_22 = BatchNormalization()
        self.activation_22 = Activation("relu")

        self.unpool_4 = UpSampling2D(pool_size)

        self.conv_23 = Convolution2D(self.conv_chan[2], (kernel, kernel), padding="same")
        self.batch_23 = BatchNormalization()
        self.activation_23 = Activation("relu")
        self.conv_24 = Convolution2D(self.conv_chan[1], (kernel, kernel), padding="same")
        self.batch_24 = BatchNormalization()
        self.activation_24 = Activation("relu")

        self.unpool_5 = UpSampling2D(pool_size)

        self.conv_25 = Convolution2D(self.conv_chan[0], (kernel, kernel), padding="same")
        self.batch_25 = BatchNormalization()
        self.activation_25 = Activation("relu")
        self.conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")
        self.batch_26 = BatchNormalization()
        self.activation_26 = Activation(output_mode)

    def call(self, inputs, training=None, mask=None):
        x = self.conv_1(inputs)
        x = self.batch_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.activation_2(x)

        # shape1 = tf.shape(x)
        x = self.pool_1(x)

        x = self.conv_3(x)
        x = self.batch_3(x)
        x = self.activation_3(x)
        x = self.conv_4(x)
        x = self.batch_4(x)
        x = self.activation_4(x)

        # shape2 = tf.shape(x)
        x = self.pool_2(x)

        x = self.conv_5(x)
        x = self.batch_5(x)
        x = self.activation_5(x)
        x = self.conv_6(x)
        x = self.batch_6(x)
        x = self.activation_6(x)
        x = self.conv_7(x)
        x = self.batch_7(x)
        x = self.activation_7(x)

        # shape3 = tf.shape(x)
        x = self.pool_3(x)

        x = self.conv_8(x)
        x = self.batch_8(x)
        x = self.activation_8(x)
        x = self.conv_9(x)
        x = self.batch_9(x)
        x = self.activation_9(x)
        x = self.conv_10(x)
        x = self.batch_10(x)
        x = self.activation_10(x)

        # shape4 = tf.shape(x)
        x = self.pool_4(x)

        x = self.conv_11(x)
        x = self.batch_11(x)
        x = self.activation_11(x)
        x = self.conv_12(x)
        x = self.batch_12(x)
        x = self.activation_12(x)
        x = self.conv_13(x)
        x = self.batch_13(x)
        x = self.activation_13(x)

        # shape5 = tf.shape(x)
        x = self.pool_5(x)

        # decoder

        x = self.unpool_1(x)
        # x = tf.slice(x, [0, 0, 0, 0], [1, shape5[1], shape5[2], self.conv_chan[12]])

        x = self.conv_14(x)
        x = self.batch_14(x)
        x = self.activation_14(x)
        x = self.conv_15(x)
        x = self.batch_15(x)
        x = self.activation_15(x)
        x = self.conv_16(x)
        x = self.batch_16(x)
        x = self.activation_16(x)

        x = self.unpool_2(x)
        # x = tf.slice(x, [0, 0, 0, 0], [1, shape4[1], shape4[2], self.conv_chan[9]])

        x = self.conv_17(x)
        x = self.batch_17(x)
        x = self.activation_17(x)
        x = self.conv_18(x)
        x = self.batch_18(x)
        x = self.activation_18(x)
        x = self.conv_19(x)
        x = self.batch_19(x)
        x = self.activation_19(x)

        x = self.unpool_3(x)
        # x = tf.slice(x, [0, 0, 0, 0], [1, shape3[1], shape3[2], self.conv_chan[6]])

        x = self.conv_20(x)
        x = self.batch_20(x)
        x = self.activation_20(x)
        x = self.conv_21(x)
        x = self.batch_21(x)
        x = self.activation_21(x)
        x = self.conv_22(x)
        x = self.batch_22(x)
        x = self.activation_22(x)

        x = self.unpool_4(x)
        # x = tf.slice(x, [0, 0, 0, 0], [1, shape2[1], shape2[2], self.conv_chan[3]])

        x = self.conv_23(x)
        x = self.batch_23(x)
        x = self.activation_23(x)
        x = self.conv_24(x)
        x = self.batch_24(x)
        x = self.activation_24(x)

        x = self.unpool_5(x)
        # x = tf.slice(x, [0, 0, 0, 0], [1, shape1[1], shape1[2], self.conv_chan[1]])

        x = self.conv_25(x)
        x = self.batch_25(x)
        x = self.activation_25(x)
        x = self.conv_26(x)
        x = self.batch_26(x)
        x = self.activation_26(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_labels": self.n_labels,
            "kernel": self.kernel,
            "pool_size": self.pool_size,
            "output_mode": self.output_mode
        })
        return config



class MaxPoolingWithArgmax2D(Layer):
    """MaxPooling for unpooling with indices.

    # References
        [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation]
        (https://arxiv.org/abs/1511.00561)

    # related code:
        https://github.com/PavlosMelissinos/enet-keras
        https://github.com/ykamikawa/SegNet
    """

    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

    def call(self, inputs, **kwargs):
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides[0], self.strides[1], 1]
        padding = self.padding.upper()
        output, argmax = nn_ops.max_pool_with_argmax(inputs, ksize, strides, padding)
        argmax = tf.cast(argmax, K.floatx())
        return output, argmax

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

    def get_config(self):
        config = super(MaxPoolingWithArgmax2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

