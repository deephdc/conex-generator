import tensorflow as tf
import numpy as np


class DataMaskStandardizer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._nan = tf.constant(np.nan, dtype=tf.float32)

    def build(self, input_shape):
        self._gen_features = input_shape[1][2]

        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        data = inputs[0]
        mask = inputs[1]

        # find value before first nan in mask
        current_batchsize = tf.shape(mask)[0]
        nan_line = tf.fill([current_batchsize, 1, self._gen_features],
                           self._nan)
        mask_with_nan_line = tf.concat([mask, nan_line], axis=1)

        nanpos = tf.math.is_nan(mask_with_nan_line)
        nanpos_shifted = tf.roll(nanpos, shift=-1, axis=1)
        position_marker = tf.math.logical_xor(nanpos, nanpos_shifted)[:,0:-1,:]

        # fill mask nan positions with last non-nanval in data
        last_val = tf.where(position_marker, data, -1.0)
        last_val = tf.math.reduce_max(last_val, axis=1)
        last_val = tf.expand_dims(last_val, axis=1)

        data = tf.where(tf.math.is_nan(mask), last_val, data)

        return data

