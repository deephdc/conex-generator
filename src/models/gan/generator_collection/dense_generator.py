import tensorflow as tf
import tensorflow.keras.layers as layers

from src.models.gan import utils


class DenseGenerator(tf.keras.layers.Layer):

    def __init__(self, depthlen=288, gen_features=8, numparticle=6,
                 overscale=10.0, expscale=False,
                 activation=tf.keras.activations.tanh, **kwargs):
        super().__init__(**kwargs)

        self.depthlen = depthlen
        self.gen_features = gen_features

        self.overscale = 10.0
        self.expscale = expscale

        self.labelmerger = utils.LabelMerger(numparticle=numparticle)

        self.activation = activation
        if self.expscale:
            last_activation = tf.keras.activations.elu
        else:
            last_activation = tf.keras.activations.sigmoid

        self.layer1 = layers.Dense(512, activation=self.activation)
        self.layer2 = layers.Dense(1024, activation=self.activation)
        self.layer3 = layers.Dense(self.depthlen * self.gen_features,
                                   activation=last_activation)

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

        if self.expscale:
            tensor = tf.math.exp(-1.0*(tensor + 1.0 - tf.math.log(self.overscale)))
        else:
            tensor = tensor * self.overscale

        return tensor

