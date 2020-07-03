import tensorflow as tf
import numpy as np

from src.models.gan import generator_collection as gencol
from src.models.gan import utils


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

        self.models = [
                gencol.DenseGenerator(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=False),

                gencol.DenseGenerator(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=True),

                gencol.OldReducedGenerator(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=False),

                gencol.OldReducedGenerator(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=True),

                gencol.DenseGeneratorNorm(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=False),

                gencol.DenseGeneratorNorm(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=True),

                gencol.OldReducedGeneratorNorm(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=False),

                gencol.OldReducedGeneratorNorm(
                    self.maxdepthlen, self.gen_features, self._numparticle,
                    expscale=True),
        ]

        self.num_model = len(self.models)

        self._ensemble_var = self.add_weight(
                name="ensemble",
                shape=(self.num_model,),
                dtype=tf.bool,
                initializer=tf.keras.initializers.Constant(True),
                trainable=False,)

        self.ensemble_weights = self.add_weight(
                name="ensemble_weights",
                shape=(self.num_model,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant([1.0]*self.num_model),
                trainable=False,)

    @property
    def ensemble(self):
        return self._ensemble_var.numpy()

    @ensemble.setter
    def ensemble(self, value):
        if isinstance(value, (int, np.integer)):
            if value == 0:
                bool_list = [True for _ in range(self.num_model)]
                self._ensemble_var.assign(bool_list)
                return

            if value > 0:
                assert np.abs(value) <= self.num_model
                bool_list = [
                        True if ii + 1 == value else False
                        for ii in range(self.num_model)
                ]
                self._ensemble_var.assign(bool_list)
                return

            if value < 0:
                bit_string = format(-value, "b")[::-1]
                bit_string += "".join(["0" for _ in range(self.num_model)])
                bool_list = [
                        True if bit_string[ii] == "1" else False
                        for ii in range(self.num_model)
                ]
                self._ensemble_var.assign(bool_list)
                return

        if isinstance(value, (list, tuple)):
            assert len(value) == self.num_model
            assert np.all([
                isinstance(val, (bool, np.bool))
                for val in value
            ])
            self._ensemble_var.assign(value)
            return

        raise TypeError("unsupported type for model selection")

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise = inputs[1:]

        # run different generators
        batchsize = label.shape[0]

        zeros = tf.zeros((batchsize,self.depthlen,self.gen_features,))
        outputs = [zeros] * self.num_model

        ensemble_weights = tf.math.log(tf.math.exp(self.ensemble_weights) + 1.0)
        used_ensemble_weight = 0.0

        for ii in range(self.num_model):
            if self._ensemble_var[ii]:
                outputs[ii] = (self.models[ii]([label,noise,]))[:,0:self.depthlen,:]

        # merge outputs
        tensor = zeros
        for ii in range(self.num_model):
            if self._ensemble_var[ii]:
                tensor += ensemble_weights[ii] * outputs[ii]
                used_ensemble_weight += ensemble_weights[ii]
        tensor /= used_ensemble_weight

        # format data
        data = self.datasplitter(tensor)
        data = self.denormalizer(data)

        return data

