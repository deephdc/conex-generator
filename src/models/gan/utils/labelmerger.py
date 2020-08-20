import tensorflow as tf


class LabelMerger(tf.keras.layers.Layer):
    """This class is used to bring label data into the right format.

    It performs onehot encoding of the particle type and logscaling for the
    energy if exponential scaling is used for the model.
    
    This class implements the tf.keras.layers.Layer interface.
    """

    def __init__(self, numparticle=6, expscale=False, epsilon=1e-25, **kwargs):
        """Construct a LabelMerger instance.

        Parameters
        ----------
        numparticle : int, optional
            Maximum number of particle types that should be generated (used
            for onehot encoding of particle types). Defaults to 6.
        expscale : bool, optional
            Flag to indicate if the model uses exponential scaling of outputs.
            If so, the energy label should be logarithmically scaled.
            Defaults to False.
        epsilon : float, optional
            Regularizing constant for the logscaling:
                log(abs(x) + epsilon)
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self._numparticle = numparticle

        self.expscale = expscale
        self.epsilon = epsilon

    @tf.function
    def call(self, inputs, training=False):
        """TensorFlow model call function.

        Parameters
        ----------
        inputs : tf.Tensor
            Label input, layout: (batch, features)
        training : bool, optional
            Flag to indicate if the layer is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : tf.Tensor
            Formated label output, layout: (batch, onehot + newfeatures)
        """
        # normalize inputs
        particle = tf.cast(inputs[:,0], tf.int32)
        particle_oh = tf.one_hot(particle, self._numparticle, dtype=tf.float32)
        energy = inputs[:,1] / 1e10
        theta = inputs[:,2] / 90.0
        phi = inputs[:,3] / 180.0

        if self.expscale:
            energy = tf.math.log(tf.math.abs(energy) + self.epsilon)

        # merge
        tensor = tf.concat(
            [
                particle_oh,
                tf.expand_dims(energy, -1),
                tf.expand_dims(theta, -1),
                tf.expand_dims(phi, -1),
            ],
            -1
        )

        return tensor

