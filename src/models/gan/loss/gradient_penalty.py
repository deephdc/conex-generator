import tensorflow as tf
import numpy as np

from src.models.gan import utils


class GradientPenalty(tf.keras.Model):
    """Calculates the gradient penalty for a given discriminator.
    
    This class takes a BaseDiscriminator instance and runs all necessary steps
    to calculate the gradient penalty for non-exponential (re-)scaled inputs.
    The penalty is only calculated on data inputs, not on labels!

    This class uses the tf.keras.Model interface.
    """

    def __init__(self, discriminator, pd_maxdata=None, ed_maxdata=None, lp_dynamic=True, **kwargs):
        """Construct the GradientPenalty for a given discriminator.

        Notice: The penalty is only calculated on data inputs, not on labels!
        
        Parameters
        ----------
        discriminator : src.models.gan.BaseDiscriminator
            A BaseDiscriminator instance for which the gradient penatly
            should be calculated.
        pd_maxdata : list, optional
            List of normalization constants for the particle distribution
            channels. Length should be 8. Defaults to None, which will try
            to read the constants from the discriminator.
            These constants are essential for the gradient penalty because
            it is calculated on the "outer" discriminator, meaning that
            normalization has to be undone in order to get reasonable
            magnitudes.
        ed_maxdata : list, optional
            List of normalization constants for the energy deposit
            channels. Length should be 9. Defaults to None, which will try
            to read the constants from the discriminator.
            These constants are essential for the gradient penalty because
            it is calculated on the "outer" discriminator, meaning that
            normalization has to be undone in order to get reasonable
            magnitudes.
        lp_dynamic : bool, optional
            Flag to indicate if the lipschitz constant should be dynamically
            adjusted for the (actual) dimensionality of the inputs (no nans)
            or if it should be a fixed constant.
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.Model.
        """
        super().__init__(**kwargs)

        self.lp_dynamic = lp_dynamic
        self.discriminator = discriminator

        if pd_maxdata is None:
            self.pd_maxdata = self.discriminator.pd_maxdata
        else:
            self.pd_maxdata = pd_maxdata

        if ed_maxdata is None:
            self.ed_maxdata = self.discriminator.ed_maxdata
        else:
            self.ed_maxdata = ed_maxdata

        self.denormalizer = utils.DataDenormalizer(self.pd_maxdata,
                                                   self.ed_maxdata)

        self.superposition_pd = utils.UniformSuperposition()
        self.superposition_ed = utils.UniformSuperposition()

    def build(self, input_shape):
        """TensorFlow model build function.

        If a fixed Lipschitz constant is used, it is calculated and stored
        here once.

        Parameters
        ----------
        input_shape
            See tf.keras.Model.build
        """
        self.ndim = np.array([
            input_shape[1][1] * input_shape[1][2], # particle distribution
            input_shape[2][1] * input_shape[2][2], # energy deposit
            input_shape[0][1],                     # label
            ],
            dtype = np.float32)

        self.lipschitz_constant = tf.sqrt(self.ndim[0] + self.ndim[1])

        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        """TensorFlow model call function.

        Parameters
        ----------
        inputs : list
            Index 0: label, layout: (batch, features)
            Index 1: real particle distribution, layout: (batch, depth, channel)
            Index 2: real energy deposit, layout: (batch, depth, channel)
            Index 3: fake particle distribution, layout: (batch, depth, channel)
            Index 4: fake energy deposit, layout: (batch, depth, channel)
        training : bool, optional
            Flag to indicate if the model is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : tf.Tensor
            Scalar gradient penalty value.
        """
        label = inputs[0]
        real = inputs[1:3]
        fake = inputs[3:]

        # superimpose particle distribution and energy deposit
        sup_pd = self.superposition_pd([real[0], fake[0],])
        sup_ed = self.superposition_ed([real[1], fake[1],])

        # discriminator forward map
        tensor = self.discriminator([label, *[sup_pd, sup_ed], *real,])

        # calculate gradients
        grads = tf.gradients(tensor, [sup_pd, sup_ed],
                             unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # undo input scaling
        grads_scale = self.denormalizer([grads[0], grads[1],])

        # calculate l2 norm of all gradients
        gradsum_pd = tf.math.reduce_sum(tf.square(grads_scale[0]), axis=[1,2])
        gradsum_ed = tf.math.reduce_sum(tf.square(grads_scale[1]), axis=[1,2])

        gradnorm = tf.sqrt(gradsum_pd + gradsum_ed + 1e-6)

        if self.lp_dynamic:
            # calculate dynamic lipschitz constant based on actual input dimension
            not_nan0 = tf.math.logical_not(tf.math.is_nan(real[0]))
            not_nan1 = tf.math.logical_not(tf.math.is_nan(real[1]))

            dims0 = tf.math.count_nonzero(not_nan0, axis=[1,2], dtype=tf.float32)
            dims1 = tf.math.count_nonzero(not_nan1, axis=[1,2], dtype=tf.float32)

            lipschitz_dynamic = tf.sqrt(dims0 + dims1)

            gradient_penalty_batch = tf.abs(gradnorm - lipschitz_dynamic)
        else:
            # use fixed lipschitz constant
            gradient_penalty_batch = tf.abs(gradnorm - self.lipschitz_constant)

        # calculate gradient penalty
        gradient_penalty = tf.math.reduce_mean(gradient_penalty_batch, axis=0)

        return gradient_penalty

