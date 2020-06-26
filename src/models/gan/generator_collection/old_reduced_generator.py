import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

from src.models.gan import utils


class OldReducedGenerator(tf.keras.layers.Layer):

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

        self._nheight = 4
        self._nwidth = int(np.ceil(self.depthlen / (2 * 2 * 2 * 2 * 2)))
        self._nfilter = 32
        
        # dense label/noise combination
        self.layer_combine1 = layers.Dense(512, activation=self.activation)
        self.layer_combine2 = layers.Dense(self._nheight*self._nwidth*self._nfilter,
                                           activation=self.activation)

        # 1st transpose then conv
        self.layer_tc1 = layers.Conv2DTranspose(64, (2,5), strides=(1,2),
                                                padding="same",
                                                activation=self.activation)
        self.layer_rc1 = layers.Conv2D(64, (2,5), padding="same",
                                       activation=self.activation)

        # 2nd transpose then conv
        self.layer_tc2 = layers.Conv2DTranspose(32, (2,10), strides=(1,2),
                                                padding="same",
                                                activation=self.activation)
        self.layer_rc2 = layers.Conv2D(32, (2,10), padding="same",
                                       activation=self.activation)

        # 3rd transpose then conv
        self.layer_tc3 = layers.Conv2DTranspose(16, (2,10), strides=(1,2),
                                                padding="same",
                                                activation=self.activation)
        self.layer_rc3 = layers.Conv2D(16, (2,10), padding="same",
                                       activation=self.activation)

        # 4th transpose then conv
        self.layer_tc4 = layers.Conv2DTranspose(32, (2,5), strides=(1,2),
                                                padding="same",
                                                activation=self.activation)
        self.layer_rc4 = layers.Conv2D(32, (2,5), padding="same",
                                       activation=self.activation)


        # 5th transpose then conv
        self.layer_tc5 = layers.Conv2DTranspose(16, (2,5), strides=(1,2),
                                                padding="same",
                                                activation=self.activation)
        self.layer_rc5 = layers.Conv2D(self.gen_features, (self._nheight, 5),
                                       padding="valid",
                                       activation=last_activation)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise = inputs[1][0]

        labeltensor = self.labelmerger(label)
        tensor = tf.concat([labeltensor, noise], -1)
        
        # dense
        tensor = self.layer_combine1(tensor)
        tensor = self.layer_combine2(tensor)

        # reshape for convs
        tensor = tf.reshape(tensor, [-1, self._nheight, self._nwidth, self._nfilter])

        # convs
        tensor = self.layer_tc1(tensor)
        tensor = self.layer_rc1(tensor)

        tensor = self.layer_tc2(tensor)
        tensor = self.layer_rc2(tensor)

        tensor = self.layer_tc3(tensor)
        tensor = self.layer_rc3(tensor)

        tensor = self.layer_tc4(tensor)
        tensor = self.layer_rc4(tensor)

        tensor = self.layer_tc5(tensor)
        tensor = tf.pad(tensor, ([0,0], [0,0], [2,2], [0,0]))
        tensor = self.layer_rc5(tensor)

        # remove dimensions of size 1
        # shape should be [None, self.depthlen, self.gen_features]
        tensor = tf.squeeze(tensor, [1])[:,0:self.depthlen,:]

        if self.expscale:
            tensor = tf.math.exp(-1.0*(tensor + 1.0 - tf.math.log(self.overscale)))
        else:
            tensor = tensor * self.overscale

        return tensor

