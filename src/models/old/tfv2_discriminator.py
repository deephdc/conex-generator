import tensorflow as tf


class Discriminator(tf.keras.Model):

    def __init__(self, maxdata, **kwargs):
        super().__init__(**kwargs)

        self.maxdata = maxdata



    @tf.function
    def call(self, inputs, training=False):
        label = tf.cast(inputs[0], tf.float32)
        data = tf.cast(inputs[1], tf.float32)

        # normalize inputs
        particle = tf.cast(label[:,0], tf.int32)
        particle_oh = tf.cast(tf.one_hot(particle, 6), tf.float32)
        energy = label[:,1] / 1e10
        theta = label[:,2] / 90.0
        phi = label[:,3] / 180.0

        labeltensor = tf.concat(
            [
                particle_oh,
                tf.expand_dims(energy, -1),
                tf.expand_dims(theta, -1),
                tf.expand_dims(phi, -1),
            ],
            -1
        )

        # normalize data:
        temp0 = tf.expand_dims(data[:,:,0] / self.maxdata[0], -1)
        temp1 = tf.expand_dims(data[:,:,1] / self.maxdata[1], -1)
        temp2 = tf.expand_dims(data[:,:,2] / self.maxdata[2], -1)
        temp3 = tf.expand_dims(data[:,:,3] / self.maxdata[3], -1)
        temp4 = tf.expand_dims(data[:,:,4] / self.maxdata[4], -1)
        temp5 = tf.expand_dims(data[:,:,5] / self.maxdata[5], -1)
        temp6 = tf.expand_dims(data[:,:,6] / self.maxdata[6], -1)
        temp7 = tf.expand_dims(data[:,:,7] / self.maxdata[7], -1)

        normdata = tf.concat(
                [temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7],
                -1)

