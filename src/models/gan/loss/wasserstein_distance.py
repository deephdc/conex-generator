import tensorflow as tf


class WassersteinDistance(tf.keras.Model):

    def __init__(self, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        real = inputs[1]
        fake = inputs[2]

        realtensor = self.discriminator((label, real, real,))
        faketensor = self.discriminator((label, fake, real,))

        realtensor = tf.math.reduce_mean(realtensor)
        faketensor = tf.math.reduce_mean(faketensor)

        return (realtensor - faketensor)

