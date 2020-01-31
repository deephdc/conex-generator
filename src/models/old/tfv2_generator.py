import tensorflow as tf
import numpy as np


class Generator(tf.keras.Model):

    def __init__(self, maxdata, **kwargs):
        super().__init__(**kwargs)

        self.maxdata = maxdata

        self.nheight = 4
        self.nwidth = 16
        self.nfilter = 64
        self.numchannels = 8

        self.layer1 = tf.keras.layers.Dense(512, activation=tf.keras.activations.tanh)
        self.layer2 = tf.keras.layers.Dense(1024, activation=tf.keras.activations.tanh)
        self.layer3 = tf.keras.layers.Dense(2048, activation=tf.keras.activations.tanh)
        self.layer4 = tf.keras.layers.Dense(4096, activation=tf.keras.activations.tanh)
        self.layer5 = tf.keras.layers.Dense(self.nheight*self.nwidth*self.nfilter, activation=tf.keras.activations.tanh)

        # 1st transpose then conv
        self.layer6 = tf.keras.layers.Conv2DTranspose(self.nfilter, (2,5), strides=(1,2), padding="same", activation=tf.keras.activations.tanh)
        self.layer7 = tf.keras.layers.Conv2D(self.nfilter//2, (2,5), padding="same", activation=tf.keras.activations.tanh)

        # 1st additional
        self.layer8 = tf.keras.layers.Conv2DTranspose(self.nfilter//2, (2,10), strides=(1,1), padding="same", activation=tf.keras.activations.tanh)
        self.layer9 = tf.keras.layers.Conv2D(self.nfilter//2, (2,10), padding="same", activation=tf.keras.activations.tanh)

        # 2nd transpose then conv
        self.layer10 = tf.keras.layers.Conv2DTranspose(self.nfilter//2, (2,10), strides=(1,4), padding="same", activation=tf.keras.activations.tanh)
        self.layer11 = tf.keras.layers.Conv2D(self.nfilter//4, (2,10), padding="same", activation=tf.keras.activations.tanh)

        # 2nd additional
        self.layer12 = tf.keras.layers.Conv2DTranspose(self.nfilter//4, (2,5), strides=(1,1), padding="same", activation=tf.keras.activations.tanh)
        self.layer13 = tf.keras.layers.Conv2D(self.nfilter//4, (2,5), padding="same", activation=tf.keras.activations.tanh)

        # 3rd transpose then conv
        self.layer14 = tf.keras.layers.Conv2DTranspose(self.nfilter//4, (2,5), strides=(1,2), padding="same", activation=tf.keras.activations.tanh)
        self.layer15 = tf.keras.layers.Conv2D(self.numchannels, (4,5), padding="valid", activation=tf.nn.sigmoid)

    
    @tf.function
    def call(self, inputs, training=False):
        label = tf.cast(inputs[0], tf.float32)
        noise = tf.cast(inputs[1], tf.float32)

        particle = tf.cast(label[:,0], tf.int32)
        particle_oh = tf.cast(tf.one_hot(particle, 6), tf.float32)
        energy = label[:,1] / 1e10
        theta = label[:,2] / 90.0
        phi = label[:,3] / 180.0

        tensor = tf.concat(
            [
                noise,
                particle_oh,
                tf.expand_dims(energy, -1),
                tf.expand_dims(theta, -1),
                tf.expand_dims(phi, -1),
            ],
            -1
        )

        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)
        tensor = self.layer5(tensor)

        tensor = tf.reshape(tensor, [tf.shape(tensor)[0], self.nheight, self.nwidth, self.nfilter])

        tensor = self.layer6(tensor)
        tensor = self.layer7(tensor)
        tensor = self.layer8(tensor)
        tensor = self.layer9(tensor)
        tensor = self.layer10(tensor)
        tensor = self.layer11(tensor)
        tensor = self.layer12(tensor)
        tensor = self.layer13(tensor)
        tensor = self.layer14(tensor)
        
        tensor = tf.pad(tensor, ([0,0], [0,0], [2,2], [0,0]))

        tensor = self.layer15(tensor)
        tensor = tf.squeeze(tensor, [1])

        #TODO denormalize with maxdata

        return tensor

