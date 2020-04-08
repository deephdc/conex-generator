import tensorflow as tf
import tensorflow.keras.layers as layers

import src.models.gan.utils as utils


class DenseGenerator(tf.keras.layers.Layer):

    def __init__(self, depthlen=288, gen_features=8, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.depthlen = depthlen
        self.gen_features = gen_features

        self.labelmerger = utils.LabelMerger(numparticle=numparticle)

        self.activation = tf.keras.activations.tanh

        self.layer1 = layers.Dense(512, activation=self.activation)
        self.layer2 = layers.Dense(1024, activation=self.activation)
        self.layer3 = layers.Dense(self.depthlen * self.gen_features,
                                   activation=tf.keras.activations.sigmoid)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise = inputs[1][0]

        labeltensor = self.labelmerger(label)
        tensor = tf.concat([labeltensor, noise], -1)

        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)

        tensor = tf.reshape(tensor, shape=[-1, self.depthlen, self.gen_features])

        return tensor * 1.5

