import tensorflow as tf
import numpy as np

from src.models.gan import utils


class GradientPenalty(tf.keras.Model):

    def __init__(self, discriminator, pd_maxdata=None, ed_maxdata=None, lp_dynamic=True, **kwargs):
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

