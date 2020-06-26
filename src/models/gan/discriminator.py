import tensorflow as tf
import numpy as np

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

        self.models = [
                discol.DenseDiscriminator(self._numparticle),
                discol.OldReducedDiscriminator(self._numparticle),
                discol.DenseDiscriminatorNorm(self._numparticle),
                discol.OldReducedDiscriminatorNorm(self._numparticle),
        ]

        self.num_model = len(self.models)

        self._ensemble_var = self.add_weight(
                name="ensemble",
                shape=(self.num_model,),
                dtype=tf.bool,
                initializer=tf.keras.initializers.Constant(True),
                trainable=False,)

        self._ensemble_weight = self.add_weight(
                name="ensemble_weight",
                shape=(self.num_model,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant([1.0]*self.num_model),
                trainable=True,)

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
        data = inputs[1:3]
        mask = inputs[3:]

        data = self.normalizer(data)
        data = self.datamerger(data)
        mask = self.datamerger(mask)
        data = self.standardizer([data, mask,])

        # run different discriminators
        batchsize = label.shape[0]

        zeros = tf.zeros((batchsize,1,))
        outputs = [zeros] * self.num_model

        ensemble_weight = tf.math.log(tf.math.exp(self._ensemble_weight) + 1.0)
        used_ensemble_weight = 0.0

        for ii in range(self.num_model):
            if self._ensemble_var[ii]:
                outputs[ii] = (self.models[ii]([label, data,]))

        # merge outputs
        tensor = zeros
        for ii in range(self.num_model):
            if self._ensemble_var[ii]:
                tensor += ensemble_weight[ii] * outputs[ii]
                used_ensemble_weight += ensemble_weight[ii]
        tensor /= used_ensemble_weight

        return tensor

