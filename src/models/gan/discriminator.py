import tensorflow as tf

from src.models.gan import discriminator_collection as discol
from src.models.gan import utils


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
        data = inputs[1:3]
        mask = inputs[3:]

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

