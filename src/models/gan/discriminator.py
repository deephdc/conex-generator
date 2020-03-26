import tensorflow as tf
import numpy as np

from .utils import DataMerger
from .utils import DataNormalizer
from .utils import LabelMerger
from .utils import UniformSuperposition
from .utils import DataMaskStandardizer


class BaseDiscriminator(tf.keras.Model):

    def __init__(self, pd_maxdata, ed_maxdata, pd_feature_list=None,
                 ed_feature_list=None, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.pd_maxdata = pd_maxdata
        self.ed_maxdata = ed_maxdata
        self._numparticle = numparticle

        self.normalizer = DataNormalizer(self.pd_maxdata, self.ed_maxdata)
        self.datamerger = DataMerger(pd_feature_list, ed_feature_list)
        self.standardizer = DataMaskStandardizer()
    
    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        data = inputs[1]
        mask = inputs[2]

        data = self.normalizer(data)
        data = self.datamerger(data)
        mask = self.datamerger(mask)
        data = self.standardizer((data, mask,))

        return data


class TPCCDiscriminator(tf.keras.Model):

    def __init__(self, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self._numparticle = numparticle

        self.labelmerger = LabelMerger(self._numparticle)

        self.nfilter = 64
        self.layer1 = tf.keras.layers.Conv2D(self.nfilter, (1,10), padding="same", activation=tf.nn.tanh)

        self.layer2 = tf.keras.layers.Conv2D(self.nfilter, (1,10), strides=(1,2), padding="same", activation=tf.nn.tanh)
        self.layer2b = tf.keras.layers.Conv2D(self.nfilter, (1,10), strides=(1,1), padding="same", activation=tf.nn.tanh)

        self.layer3 = tf.keras.layers.Conv2D(self.nfilter, (1,6), strides=(1,2), padding="same", activation=tf.nn.tanh)
        self.layer3b = tf.keras.layers.Conv2D(self.nfilter, (1,6), strides=(1,1), padding="same", activation=tf.nn.tanh)

        self.layer4 = tf.keras.layers.Conv2D(self.nfilter, (1,3), strides=(1,2), padding="same", activation=tf.nn.tanh)
        self.layer4b = tf.keras.layers.Conv2D(self.nfilter, (1,3), strides=(1,1), padding="same", activation=tf.nn.tanh)

        self.layer5 = tf.keras.layers.Flatten()
        self.layer6 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)
        self.layer6b = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        data = inputs[1]

        labeltensor= self.labelmerger(label)
        # TODO add dense layers for shape adjustment to data here

        tensor = tf.expand_dims(inputs, 1)
        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer2b(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer3b(tensor)
        tensor = self.layer4(tensor)
        tensor = self.layer4b(tensor)
        tensor = self.layer5(tensor)
        tensor = self.layer6(tensor)
        tensor = self.layer6b(tensor)

        return tensor




class WassersteinDistance(tf.keras.Model):

    def __init__(self, discriminator : BaseDiscriminator, **kwargs):
        super().__init__(**kwargs)

        self.discriminator = discriminator

    @tf.function
    def call(self, inputs, training=False):
        label = tf.cast(inputs[0], tf.float32)
        realdata = tf.cast(inputs[1], tf.float32)
        fakedata = tf.cast(inputs[2], tf.float32)

        realtensor = tf.reduce_mean(self.discriminator((label, realdata,)))
        faketensor = tf.reduce_mean(self.discriminator((label, fakedata,)))

        return (realtensor - faketensor)


class GradientPenalty(tf.keras.Model):

    def __init__(self, discriminator : BaseDiscriminator, **kwargs):
        super().__init__(**kwargs)

        self.discriminator = discriminator
        self.superposition = Superposition()

    @tf.function
    def call(self, inputs, training=False):
        label = tf.cast(inputs[0], tf.float32)
        realdata = tf.cast(inputs[1], tf.float32)
        fakedata = tf.cast(inputs[2], tf.float32)

        realmerged = self.discriminator.labelmerger((label, realdata,))
        fakemerged = self.discriminator.labelmerger((label, fakedata,))

        sup = self.superposition((realmerged, fakemerged,))
        tensor = self.discriminator.downstream(sup)

        gradnorm = tf.sqrt(tf.math.reduce_sum(tf.square(tf.gradients(tensor, sup)[0]), axis=[1,2]))
        gradient_penalty = tf.reduce_mean((gradnorm - 1)**2, axis=0)

        return gradient_penalty

