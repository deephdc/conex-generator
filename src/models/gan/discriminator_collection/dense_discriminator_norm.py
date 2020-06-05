import tensorflow as tf
import tensorflow.keras.layers as layers

from src.models.gan import utils


class DenseDiscriminatorNorm(tf.keras.layers.Layer):

    def __init__(self, numparticle=6,
                 activation=tf.keras.activations.tanh, **kwargs):
        super().__init__(**kwargs)

        self._numparticle = numparticle

        self.labelmerger = utils.LabelMerger(self._numparticle)

        self.activation = activation

        self.layer_label_flatten = layers.Flatten()
        self.layer_data_flatten = layers.Flatten()

        self.layer_dense1 = layers.Dense(1024, activation=self.activation)
        self.layer_dense1_norm = layers.LayerNormalization(epsilon=1e-6)
        self.layer_dense2 = layers.Dense(512,  activation=self.activation)
        self.layer_dense2_norm = layers.LayerNormalization(epsilon=1e-6)
        self.layer_dense3 = layers.Dense(256,  activation=self.activation)
        self.layer_dense3_norm = layers.LayerNormalization(epsilon=1e-6)
        self.layer_dense4 = layers.Dense(1,    activation=None)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        data = inputs[1]

        labeltensor= self.labelmerger(label)
        labeltensor = self.layer_label_flatten(labeltensor)

        datatensor = self.layer_data_flatten(data)

        # combine label and data
        tensor = tf.concat([datatensor, labeltensor], -1)

        tensor = self.layer_dense1(tensor)
        tensor = self.layer_dense1_norm(tensor)
        tensor = self.layer_dense2(tensor)
        tensor = self.layer_dense2_norm(tensor)
        tensor = self.layer_dense3(tensor)
        tensor = self.layer_dense3_norm(tensor)
        tensor = self.layer_dense4(tensor)

        return tensor

