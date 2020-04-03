import tensorflow as tf

import src.models.gan.generator_collection as gencol
import src.models.gan.discriminator_collection as discol 
import src.models.gan.utils as utils


def get_maxdepthlen():
    return 2*2*2*2*2*3*3 # 288


class BaseGenerator(tf.keras.Model):

    def __init__(self, depthlen, pd_maxdata, ed_maxdata, pd_feature_list=None,
                 ed_feature_list=None, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.depthlen = depthlen
        self.maxdepthlen = get_maxdepthlen()
        self.pd_maxdata = pd_maxdata
        self.ed_maxdata = ed_maxdata
        self._numparticle = numparticle

        self.datasplitter = utils.DataSplitter(pd_feature_list,
                                               ed_feature_list)

        self.denormalizer = utils.DataDenormalizer(self.pd_maxdata,
                                                   self.ed_maxdata)

        self.gen_features = self.datasplitter.gen_features

        self.dense_generator = gencol.DenseGenerator(self.maxdepthlen,
                                                     self.gen_features,
                                                     self._numparticle)

        self.oldr_generator = gencol.OldReducedGenerator(self.maxdepthlen,
                                                         self.gen_features,
                                                         self._numparticle)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise = inputs[1]

        # run different generators
        output1 = self.dense_generator((label,noise,))
        output2 = self.oldr_generator((label,noise,))

        # merge outputs
        tensor = output1 + output2

        # format data
        tensor = tensor[:,0:self.depthlen,:]
        tensor = self.datasplitter(tensor)
        tensor = self.denormalizer(tensor)

        return tensor

