import tensorflow as tf
import numpy as np


class DataMaskStandardizer(tf.keras.layers.Layer):
    """DataMaskStandardizer takes data and replaces values according to a given mask.

    Because longitudinal profiles have varying length (due to different
    zeniths), this class removes the undefined values (nans) in a fixed size
    tensor and replaces them with the last known value (i.e. the last valid
    value in a given channel).
    This is neccessary in order to
        1) stop gradients in the unphysical range (below observation level)
        2) remove highly non-linear jumping to undefined values.
    The mask itself flags undefined values (which should be altered) by nans,
    such that real data instances can be used as masking tensors.

    This class implements the tf.keras.layers.Layer interface.
    """

    def __init__(self, **kwargs):
        """Construct a DataMaskStandardizer instance.

        Paramters
        ---------
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)

        self._nan = tf.constant(np.nan, dtype=tf.float32)

    def build(self, input_shape):
        """TensorFlow model build function.

        Derives the number of features/channels from the input shape.

        Parameters
        ----------
        input_shape
            See tf.keras.layers.Layer.build.
        """
        self._gen_features = input_shape[1][2]

        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        """TensorFlow model call function.

        Parameters
        ----------
        inputs : list
            Index 0: merged data, layout: (batch, depth, net-channels)
            Index 1: mask, layout: (batch, depth, net-channels)
        training : bool, optional
            Flag to indicate if the layer is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : tf.Tensor
            Standardized data tensor, layout (batch, depth, net-channels)
        """
        data = inputs[0]
        mask = inputs[1]

        # find value before first nan in mask
        current_batchsize = tf.shape(mask)[0]
        nan_line = tf.fill([current_batchsize, 1, self._gen_features],
                           self._nan)
        mask_with_nan_line = tf.concat([mask, nan_line], axis=1)

        nanpos = tf.math.is_nan(mask_with_nan_line)
        nanpos_shifted = tf.roll(nanpos, shift=-1, axis=1)
        position_marker = tf.math.logical_xor(nanpos, nanpos_shifted)[:,0:-1,:]

        # fill mask nan positions with last non-nanval in data
        last_val = tf.where(position_marker, data, -1.0)
        last_val = tf.math.reduce_max(last_val, axis=1)
        last_val = tf.expand_dims(last_val, axis=1)

        data = tf.where(tf.math.is_nan(mask), last_val, data)

        return data

