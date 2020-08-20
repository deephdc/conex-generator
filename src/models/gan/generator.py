import tensorflow as tf
import numpy as np

from src.models.gan import generator_collection as gencol
from src.models.gan import utils


def get_maxdepthlen():
    return 2*2*2*2*2*3*3 # 288


class BaseGenerator(tf.keras.Model):
    """This is the base generator class, which calls specific subgenerators.

    BaseGenerator is a wrapper class that implements ensemble handling and
    is responsible for calling the appropriate generator types. It uses the
    tf.keras.Model interface.
    """

    def __init__(self, depthlen, pd_maxdata, ed_maxdata, pd_feature_list=None,
                 ed_feature_list=None, numparticle=6, **kwargs):
        """Construct a BaseGenerator.

        Parameters
        ----------
        depthlen : int
            Lenght in depth bins of the generated data.
        pd_maxdata : list
            List of normalization constants for the particle distribution
            channels. Length should be 8.
        ed_maxdata : list
            List of normalization constants for the energy deposit
            channels. Length should be 9.
        pd_feature_list : list, optional
            List of indices for particle distribution channels that should be
            generated. Defaults to None, which will use [0,1,2,3,4,5,6]
            (everything except nuclei).
        ed_feature_list : list, optional
            List of indices for energy deposit channels that should be
            generated. Defaults to None, which will use [8] (nothing except
            sum/total energy deposit).
        numparticle : int, optional
            Maximum number of particle types that should be generated (used
            for onehot encoding of particle types).
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.Model.
        """
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
        """Ensemble status as a boolean numpy array. True means enabled."""
        return self._ensemble_var.numpy()

    @ensemble.setter
    def ensemble(self, value):
        """Set the ensemble status.

        This setter function will raise an exception if an unsupported input
        type is given.

        Paramters
        ---------
        value : int, list, tuple, np.ndarray
            Indicator value of the desired ensemble status. Handling will
            depend on the input type.

            integer:
                == 0 -> enable the whole ensemble.
                 > 0 -> enable this index and disable all the others.
                 < 0 -> the absolute value is treated as a boolean bit mask.

            list, tuple, np.ndarray
                Sequence whose length should match the number of models in
                the ensemble. Values have to be boolean. True means enabled.
        """
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

        if isinstance(value, (list, tuple, np.ndarray)):
            assert len(value) == self.num_model
            assert np.all([
                isinstance(val, (bool, np.bool, np.bool_))
                for val in value
            ])
            if not np.any(value):
                bool_list = [True for _ in range(self.num_model)]
                self._ensemble_var.assign(bool_list)
            else:
                self._ensemble_var.assign(value)
            return

        raise TypeError("unsupported type for model selection")

    @tf.function
    def call(self, inputs, training=False):
        """TensorFlow model call function.

        Parameters
        ----------
        inputs : list
            Index 0: label, layout: (batch, features)
            Index 1-end: noise vectors, layout: (type, batch, dimensions)
        training : bool
            Flag to indicate if the model is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : list
            Index 0: tf.Tensor, particle distribution
            Index 1: tf.Tensor, energy deposit
        """
        label = inputs[0]
        noise = inputs[1:]

        # run different generators
        current_batchsize = tf.shape(label)[0]

        zeros = tf.zeros((current_batchsize,self.depthlen,self.gen_features,))
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

