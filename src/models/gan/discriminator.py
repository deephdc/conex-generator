import tensorflow as tf

import src.models.gan.discriminator_collection as discol
import src.models.gan.utils as utils


class BaseDiscriminator(tf.keras.Model):

    def __init__(self, pd_maxdata, ed_maxdata, pd_feature_list=None,
                 ed_feature_list=None, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.pd_maxdata = pd_maxdata
        self.ed_maxdata = ed_maxdata
        self._numparticle = numparticle

        self.normalizer = utils.DataNormalizer(self.pd_maxdata, self.ed_maxdata)
        self.datamerger = utils.DataMerger(pd_feature_list, ed_feature_list)
        self.standardizer = utils.DataMaskStandardizer()

        self.dense_discriminator = discol.DenseDiscriminator(self._numparticle)
        self.oldr_discriminator = discol.OldReducedDiscriminator(self._numparticle)
    
    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        data = inputs[1]
        mask = inputs[2]

        data = self.normalizer(data)
        data = self.datamerger(data)
        mask = self.datamerger(mask)
        data = self.standardizer((data, mask,))


        # run different generators
        output1 = self.dense_discriminator((label, data,))
        output2 = self.oldr_discriminator((label, data,))

        # merge outputs
        tensor = output1 + output2

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

