import tensorflow as tf
import numpy as np


class UniformSuperposition(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == len(input_shape[1])
        for ii in range(len(input_shape[0])):
            assert input_shape[0][ii] == input_shape[1][ii]

        self._epsilon_shape = np.ones_like(input_shape[0])
        self._epsilon_shape[0] = -1
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        data0 = inputs[0]
        data1 = inputs[1]

        epsilon_shape = tf.where(self._epsilon_shape == -1, tf.shape(data0)[0], self._epsilon_shape)
        epsilon = tf.random.uniform(shape=epsilon_shape , dtype=tf.float32)

        tensor = epsilon * data0 + (1 - epsilon) * data1
        return tensor

