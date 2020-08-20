import tensorflow as tf


class WassersteinDistance(tf.keras.Model):
    """Calculates the Wasserstein distance for a given discriminator.
    
    This class takes a BaseDiscriminator instance and runs all necessary steps
    to calculate the Wasserstein distance.

    This class uses the tf.keras.Model interface.
    """

    def __init__(self, discriminator, **kwargs):
        """Construct the WassersteinDistance for a given discriminator.

        Parameters
        ----------
        discriminator : src.models.gan.BaseDiscriminator
            A BaseDiscriminator instance for which the Wasserstein distance
            should be calculated.
        **kwargs
            Additional keyword arguments that are passed down to the
            constructor of tf.keras.Model.
        """
        super().__init__(**kwargs)
        self.discriminator = discriminator

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
        training : bool
            Flag to indicate if the model is run in training mode or not.
            Defaults to False.

        Returns
        -------
        tensor : tf.Tensor
            Scalar Wasserstein distance value.
        """
        label = inputs[0]
        real = inputs[1:3]
        fake = inputs[3:]

        realtensor = self.discriminator([label, *real, *real,])
        faketensor = self.discriminator([label, *fake, *real,])

        realtensor = tf.math.reduce_mean(realtensor)
        faketensor = tf.math.reduce_mean(faketensor)

        return (realtensor - faketensor)

