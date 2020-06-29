import tensorflow as tf

import src.models.gan as gan


class ModelBuilder():

    def __init__(self, databuilder):
        self.databuilder = databuilder

        # constructions
        self.generator : tf.keras.Model = None
        self.discriminator : tf.keras.Model = None
        self.wasserstein_distance : tf.keras.Model = None
        self.gradient_penalty : tf.keras.Model = None

        self.optimizer_generator : tf.keras.optimizers.Optimizer = None
        self.optimizer_discriminator : tf.keras.optimizers.Optimizer = None

        # default parameters
        self.pd_feature_list = None
        self.ed_feature_list = None
        self.numparticle = 6

    def build(self, **kwargs):
        self.generator : tf.keras.Model = gan.BaseGenerator(
                depthlen=self.databuilder.depthlen,
                pd_maxdata=self.databuilder.pd_maxdata,
                ed_maxdata=self.databuilder.ed_maxdata,
                pd_feature_list=self.pd_feature_list,
                ed_feature_list=self.ed_feature_list,
                numparticle=self.numparticle,
                **kwargs)

        self.discriminator : tf.keras.Model = gan.BaseDiscriminator(
                pd_maxdata=self.databuilder.pd_maxdata,
                ed_maxdata=self.databuilder.ed_maxdata,
                pd_feature_list=self.pd_feature_list,
                ed_feature_list=self.ed_feature_list,
                numparticle=self.numparticle,
                **kwargs)

        self.wasserstein_distance : tf.keras.Model = gan.loss.WassersteinDistance(
                discriminator=self.discriminator,)

        self.gradient_penalty : tf.keras.Model = gan.loss.GradientPenalty(
                discriminator=self.discriminator,
                pd_maxdata=self.databuilder.pd_maxdata,
                ed_maxdata=self.databuilder.ed_maxdata,)

        # init once
        for label, real, noise in self.databuilder.dataset.take(1):
            out1 = self.generator([label, *noise,])
            out2 = self.discriminator([label, *real, *real,])
            out3 = self.wasserstein_distance([label, *real, *out1,])
            out4 = self.gradient_penalty([label, *real, *out1,])

        return self

