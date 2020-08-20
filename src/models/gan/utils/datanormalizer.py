import tensorflow as tf


class DataNormalizer(tf.keras.layers.Layer):
    """DataNormalizer normalizes the input/output profiles to a range of [0, 1].

    Division by zero is replaced by passing the input without normalization.

    This class implements the tf.keras.layers.Layer interface.
    """

    def __init__(self, pd_maxdata, ed_maxdata, **kwargs):
        """Construct a DataNormalizer instance.

        Paramters
        ---------
        pd_maxdata : list
            List of normalization constants for the particle distribution
            channels. Length should be 8.
        ed_maxdata : list
            List of normalization constants for the energy deposit
            channels. Length should be 9.
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)

        self._pd_maxdata = tf.constant(pd_maxdata, dtype=tf.float32)
        self._ed_maxdata = tf.constant(ed_maxdata, dtype=tf.float32)

    @tf.function
    def call(self, inputs, training=False):
        """TensorFlow model call function.

        Parameters
        ----------
        inputs : list
            Index 0: particle distribution, layout: (batch, depth, channel)
            Index 1: energy deposit, layout: (batch, depth, channel)
        training : bool, optional
            Flag to indicate if the layer is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : list
            Normalized data tensors, layout (batch, depth, channels)
            Index 0: particle distribution
            Index 1: energy deposit
        """
        pd = inputs[0]
        ed = inputs[1]

        pd_maxdata = tf.where(self._pd_maxdata == 0.0, 1.0, self._pd_maxdata)
        ed_maxdata = tf.where(self._ed_maxdata == 0.0, 1.0, self._ed_maxdata)

        pd = pd / pd_maxdata
        ed = ed / ed_maxdata

        return [pd, ed]


class DataDenormalizer(tf.keras.layers.Layer):
    """DataDenormalizer restores the input/output profiles to their original range.

    Multiplication by zero is replaced by passing the input without denormalization.

    This class implements the tf.keras.layers.Layer interface.
    """

    def __init__(self, pd_maxdata, ed_maxdata, **kwargs):
        """Construct a DataDenormalizer instance.

        Paramters
        ---------
        pd_maxdata : list
            List of normalization constants for the particle distribution
            channels. Length should be 8.
        ed_maxdata : list
            List of normalization constants for the energy deposit
            channels. Length should be 9.
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)

        self._pd_maxdata = tf.constant(pd_maxdata, dtype=tf.float32)
        self._ed_maxdata = tf.constant(ed_maxdata, dtype=tf.float32)

    @tf.function
    def call(self, inputs, training=False):
        """TensorFlow model call function.

        Parameters
        ----------
        inputs : list
            Index 0: particle distribution, layout: (batch, depth, channel)
            Index 1: energy deposit, layout: (batch, depth, channel)
        training : bool, optional
            Flag to indicate if the layer is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : list
            Denormalized data tensors, layout (batch, depth, channels)
            Index 0: particle distribution
            Index 1: energy deposit
        """
        pd = inputs[0]
        ed = inputs[1]

        pd_maxdata = tf.where(self._pd_maxdata == 0.0, 1.0, self._pd_maxdata)
        ed_maxdata = tf.where(self._ed_maxdata == 0.0, 1.0, self._ed_maxdata)

        pd = pd * pd_maxdata
        ed = ed * ed_maxdata

        return [pd, ed]

