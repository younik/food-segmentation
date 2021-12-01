from tensorflow.keras.layers import Input, Activation, Concatenate
from tensorflow.keras.layers import Convolution2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D,  UpSampling2D,


def UNet(n_categories,
         input_shape=(256, 256, 3),
         kernel=3,
         pool_size=(2, 2),
         output_mode='sigmoid'):
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    batch_1 = BatchNormalization()(conv_1)
    activation_1 = Activation("relu")(batch_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(activation_1)
    batch_2 = BatchNormalization()(conv_2)
    activation_2 = Activation("relu")(batch_2)

    pool_1 = MaxPooling2D(pool_size)(activation_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    batch_3 = BatchNormalization()(conv_3)
    activation_3 = Activation("relu")(batch_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(activation_3)
    batch_4 = BatchNormalization()(conv_4)
    activation_4 = Activation("relu")(batch_4)

    pool_2 = MaxPooling2D(pool_size)(activation_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    batch_5 = BatchNormalization()(conv_5)
    activation_5 = Activation("relu")(batch_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(activation_5)
    batch_6 = BatchNormalization()(conv_6)
    activation_6 = Activation("relu")(batch_6)

    pool_3 = MaxPooling2D(pool_size)(activation_6)

    conv_7 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    batch_7 = BatchNormalization()(conv_7)
    activation_7 = Activation("relu")(batch_7)
    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(activation_7)
    batch_8 = BatchNormalization()(conv_8)
    activation_8 = Activation("relu")(batch_8)

    pool_4 = MaxPooling2D(pool_size)(activation_8)

    conv_9 = Convolution2D(1024, (kernel, kernel), padding="same")(pool_4)
    batch_9 = BatchNormalization()(conv_9)
    activation_9 = Activation("relu")(batch_9)
    conv_10 = Convolution2D(1024, (kernel, kernel), padding="same")(activation_9)
    batch_10 = BatchNormalization()(conv_10)
    activation_10 = Activation("relu")(batch_10)

    # decoder

    unpool_1 = UpSampling2D(pool_size)(activation_10)

    conv_11 = Convolution2D(512, pool_size, padding="same")(unpool_1)
    batch_11 = BatchNormalization()(conv_11)
    # activation_11 = Activation("relu")(batch_11)
    merge_11 = Concatenate(axis=-1)([activation_8, batch_11])
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(merge_11)
    batch_12 = BatchNormalization()(conv_12)
    activation_12 = Activation("relu")(batch_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(activation_12)
    batch_13 = BatchNormalization()(conv_13)
    activation_13 = Activation("relu")(batch_13)

    unpool_2 = UpSampling2D(pool_size)(activation_13)

    conv_14 = Convolution2D(256, pool_size, padding="same")(unpool_2)
    batch_14 = BatchNormalization()(conv_14)
    # activation_14 = Activation("relu")(batch_14)
    merge_14 = Concatenate(axis=-1)([activation_6, batch_14])
    conv_15 = Convolution2D(256, (kernel, kernel), padding="same")(merge_14)
    batch_15 = BatchNormalization()(conv_15)
    activation_15 = Activation("relu")(batch_15)
    conv_16 = Convolution2D(256, (kernel, kernel), padding="same")(activation_15)
    batch_16 = BatchNormalization()(conv_16)
    activation_16 = Activation("relu")(batch_16)

    unpool_3 = UpSampling2D(pool_size)(activation_16)

    conv_17 = Convolution2D(128, pool_size, padding="same")(unpool_3)
    batch_17 = BatchNormalization()(conv_17)
    # activation_17 = Activation("relu")(batch_17)
    merge_17 = Concatenate(axis=-1)([activation_4, batch_17])
    conv_18 = Convolution2D(128, (kernel, kernel), padding="same")(merge_17)
    batch_18 = BatchNormalization()(conv_18)
    activation_18 = Activation("relu")(batch_18)
    conv_19 = Convolution2D(128, (kernel, kernel), padding="same")(activation_18)
    batch_19 = BatchNormalization()(conv_19)
    activation_19 = Activation("relu")(batch_19)

    unpool_4 = UpSampling2D(pool_size)(activation_19)

    conv_20 = Convolution2D(64, pool_size, padding="same")(unpool_4)
    batch_20 = BatchNormalization()(conv_20)
    # activation_20 = Activation("relu")
    merge_20 = Concatenate(axis=-1)([activation_2, batch_20])
    conv_21 = Convolution2D(64, (kernel, kernel), padding="same")(merge_20)
    batch_21 = BatchNormalization()(conv_21)
    activation_21 = Activation("relu")(batch_21)
    conv_22 = Convolution2D(64, (kernel, kernel), padding="same")(activation_21)
    batch_22 = BatchNormalization()(conv_22)
    activation_22 = Activation("relu")(batch_22)

    conv_23 = Convolution2D(n_categories, (1, 1), padding="valid")(activation_22)
    outputs = Activation(output_mode)(conv_23)

    model = Model(inputs=inputs, outputs=outputs, name='U-Net')

    return model