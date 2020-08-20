import tensorflow as tf
import numpy as np


class UniformSuperposition(tf.keras.layers.Layer):
    """Build a superposition of real and fake data with random, uniform weights.

    output = e * real + (1 - e) * fake
    with e ~ U(0,1)

    It is used for the gradient penalty calculation.
    
    This class implements the tf.keras.layers.Layer interface.
    """

    def __init__(self, **kwargs):
        """Construct a UniformSuperposition.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)

    def build(self, input_shape):
        """TensorFlow model build function.

        Performs shape checks and infers the shape of the random weights.

        Parameters
        ----------
        input_shape
            See tf.keras.layers.Layer.build.
        """
        assert len(input_shape[0]) == len(input_shape[1])
        for ii in range(len(input_shape[0])):
            assert input_shape[0][ii] == input_shape[1][ii]

        self._epsilon_shape = np.ones_like(input_shape[0])
        self._epsilon_shape[0] = -1
        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        """TensorFlow model call function.

        Parameters
        ----------
        inputs : list
            Index 0: real data, layout: (batch, depth, channel)
            Index 0: fake data, layout: (batch, depth, channel)
        training : bool, optional
            Flag to indicate if the layer is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : tf.Tensor
            Superpostion tensor, layout: (batch, depth, channel)
        """
        data0 = inputs[0]
        data1 = inputs[1]

        epsilon_shape = tf.where(self._epsilon_shape == -1, tf.shape(data0)[0], self._epsilon_shape)
        epsilon = tf.random.uniform(shape=epsilon_shape , dtype=tf.float32)

        tensor = epsilon * data0 + (1 - epsilon) * data1
        return tensor

