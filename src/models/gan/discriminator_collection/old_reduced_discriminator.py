import tensorflow as tf
import tensorflow.keras.layers as layers

from src.models.gan import utils


class OldReducedDiscriminator(tf.keras.layers.Layer):

    def __init__(self, numparticle=6,
                 activation=tf.keras.activations.tanh, **kwargs):
        super().__init__(**kwargs)

        self._numparticle = numparticle

        self.labelmerger = utils.LabelMerger(self._numparticle)

        self.activation = activation

        # transform label to data format
        self.layer_combine_flatten = layers.Flatten()
        self.layer_combine1 = layers.Dense(256, activation=self.activation)

        # downstream
        self.layer_conv1 = layers.Conv1D(64, 16, padding="same",
                                         activation=self.activation)
        self.layer_conv2 = layers.Conv1D(56, 14, padding="same",
                                         activation=self.activation)
        self.layer_conv3 = layers.Conv1D(48, 12, padding="same",
                                         activation=self.activation)
        self.layer_conv4 = layers.Conv1D(40, 10, padding="same",
                                         activation=self.activation)
        self.layer_conv5 = layers.Conv1D(32, 8,  padding="same",
                                         activation=self.activation)
        self.layer_conv6 = layers.Conv1D(24, 6,  padding="same",
                                         activation=self.activation)
        self.layer_conv7 = layers.Conv1D(16, 4,  padding="same",
                                         activation=self.activation)

        self.layer_end_flatten = layers.Flatten()
        self.layer_end_dense1 = layers.Dense(256, activation=self.activation)
        self.layer_end_dense2 = layers.Dense(1,   activation=None)

    def build(self, input_shape):
        self._depthlen = input_shape[1][1]

        # transform label to data format
        self.layer_combine2 = layers.Dense(self._depthlen*2,
                                           activation=self.activation)
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        data = inputs[1]

        labeltensor= self.labelmerger(label)
        labeltensor = self.layer_combine_flatten(labeltensor)
        labeltensor = self.layer_combine1(labeltensor)
        labeltensor = self.layer_combine2(labeltensor)
        labeltensor = tf.reshape(labeltensor, [-1, self._depthlen, 2])

        # combine label and data
        tensor = tf.concat([data, labeltensor], -1)

        tensor = self.layer_conv1(tensor)
        tensor = self.layer_conv2(tensor)
        tensor = self.layer_conv3(tensor)
        tensor = self.layer_conv4(tensor)
        tensor = self.layer_conv5(tensor)
        tensor = self.layer_conv6(tensor)
        tensor = self.layer_conv7(tensor)

        tensor = self.layer_end_flatten(tensor)
        tensor = self.layer_end_dense1(tensor)
        tensor = self.layer_end_dense2(tensor)

        return tensor

