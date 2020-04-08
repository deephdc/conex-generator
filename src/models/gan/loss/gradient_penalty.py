import tensorflow as tf
import numpy as np

import src.models.gan as gan
import src.models.gan.utils as utils


class GradientPenalty(tf.keras.Model):

    def __init__(self, discriminator : gan.BaseDiscriminator, **kwargs):
        super().__init__(**kwargs)

        self.discriminator = discriminator

        self.superposition0 = utils.UniformSuperposition()
        self.superposition1 = utils.UniformSuperposition()

    def build(self, input_shape):
        self.ndim = np.array([
            input_shape[1][0][1] * input_shape[1][0][2], # particle distribution
            input_shape[1][1][1] * input_shape[1][1][2], # energy deposit
            input_shape[0][1],                           # label
            ],
            dtype = np.float32)

        self.lipschitz_constant = tf.sqrt(self.ndim[0] + self.ndim[1])

        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        real = inputs[1]
        fake = inputs[2]

        # superimpose pd and ed
        sup0 = self.superposition0(real[0], fake[0]) # particle distribution
        sup1 = self.superposition1(real[1], fake[1]) # energy deposit

        # discriminator forward map
        tensor = self.discriminator((label, (sup0, sup1), real,))

        # calculate l2 norm of gradient
        gradients = tf.gradients(tensor, [sup0, sup1],
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)

        gradsum0 = tf.math.reduce_sum(tf.square(gradients[0]), axis=[1,2])
        gradsum1 = tf.math.reduce_sum(tf.square(gradients[1]), axis=[1,2])
        gradnorm = tf.sqrt(gradsum0 + gradsum1)

        # calculate dynamic lipschitz constant based actual input dimension
        not_nan0 = tf.math.logical_not(tf.math.is_nan(real[0]))
        not_nan1 = tf.math.logical_not(tf.math.is_nan(real[1]))

        dims0 = tf.math.count_nonzero(not_nan0, axis=[1,2])
        dims1 = tf.math.count_nonzero(not_nan1, axis=[1,2])

        lipschitz_dynamic = tf.sqrt(dims0 + dims1)

        # calculate gradient penalty
        gradient_penalty_batch = tf.abs(gradnorm - lipschitz_dynamic)
        gradient_penalty = tf.math.reduce_mean(gradient_penalty_batch, axis=0)

        return gradient_penalty

