from tensorflow.keras.utils import Sequence
import tensorflow as tf
import tensorflow_addons as tfa
import math
import cv2
import numpy as np
from sklearn.utils import shuffle


class FoodSequence(Sequence):

    def __init__(self, images, annotations, n_cat, img_size=(512, 512), batch_size=32, data_augmentation=True,
                 folder='train'):
        self.images = shuffle(images)
        self.annotations = annotations
        self.batch_size = batch_size
        self.n_cat = n_cat
        self.img_size = img_size
        self.folder = folder
        self.data_augmentation = data_augmentation

    def __len__(self):
        return math.floor(len(self.images) / self.batch_size)

    def on_epoch_end(self):
        self.images = shuffle(self.images)

    def __getitem__(self, idx):
        ids = self.images.index.values[idx * self.batch_size: (idx + 1) * self.batch_size]
        file_names = self.images.loc[ids]['file_name']
        imgs = [cv2.resize(cv2.imread(self.folder + '/images/' + f_name), dsize=self.img_size) / 255
                for f_name in file_names]
        desired_outs = []

        for i in range(self.batch_size):
            segmentations = self.annotations[ids[i]]
            img_row = self.images.loc[ids[i]]
            img_shape = (img_row['height'], img_row['width'])
            desired_outs.append(self._desired_output(img_shape, segmentations))

        if self.data_augmentation:
            return self._augment_data(np.array(imgs), np.array(desired_outs))
        else:
            return np.array(imgs), np.array(desired_outs, dtype=bool)

    @staticmethod
    def _augment_data(images, masks):
        if tf.random.uniform(()) > 0.5:
            images = tf.image.flip_left_right(images)
            masks = tf.image.flip_left_right(masks)
        if tf.random.uniform(()) > 0.5:
            images = tf.image.flip_up_down(images)
            masks = tf.image.flip_up_down(masks)

        images = tf.image.adjust_contrast(images, tf.random.uniform(())*0.8 + 0.6)

        angle = tf.random.uniform(())*60 - 30
        images = tfa.image.rotate(images, angle)
        masks = tfa.image.rotate(masks, angle)

        scale = tf.random.uniform(())*0.3 - 0.3
        images = tf.keras.layers.experimental.preprocessing.RandomZoom(scale)(images)
        masks = tf.keras.layers.experimental.preprocessing.RandomZoom(scale, interpolation="nearest")(masks)

        return images, np.asarray(masks, dtype=bool)

    def _desired_output(self, img_shape, segmentations):
        desired_output = np.zeros((self.img_size[0], self.img_size[1], self.n_cat), dtype=np.int32)

        for k in segmentations:
            t = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.int32)
            for seg in segmentations[k]:
                s = np.array(seg, dtype=np.int32).reshape((-1, 2))
                t = cv2.resize(cv2.fillPoly(t, [s], 255), dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            desired_output[:, :, k] = t.squeeze()

        return desired_output / 255


