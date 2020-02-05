import tensorflow as tf


class LabelMerger(tf.keras.Model):

    def __init__(self, maxdata, **kwargs):
        super().__init__(**kwargs)

        self.maxdata = maxdata

        self.label_layer1 = tf.keras.layers.Dense(4096, activation=tf.nn.tanh)
        self.label_layer2 = tf.keras.layers.Dense(8192, activation=tf.nn.tanh)

    def build(self, input_shape):
        self.depthlen = input_shape[1][1]
        self.numchannels = input_shape[1][2]

        self.label_layer3 = tf.keras.layers.Dense(self.depthlen*self.numchannels, activation=tf.nn.tanh)

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

        labeltensor = self.label_layer1(labeltensor)
        labeltensor = self.label_layer2(labeltensor)
        labeltensor = self.label_layer3(labeltensor)
        labeltensor = tf.reshape(labeltensor, [tf.shape(labeltensor)[0], self.depthlen, self.numchannels])

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
                -1
        )

        tensor = tf.concat(
                [
                    normdata,
                    labeltensor,
                ],
                -1
        )

        return tensor


class Superposition(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, inputs, training=False):
        realdata = tf.cast(inputs[0], tf.float32)
        fakedata = tf.cast(inputs[1], tf.float32)

        epsilon = tf.random.uniform(shape = [tf.shape(realdata)[0], 1, 1], dtype=tf.float32)
        tensor = epsilon * realdata + (1 - epsilon) * fakedata

        return tensor


class Downstream(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.nfilter = 64

        self.layer1 = tf.keras.layers.Conv2D(self.nfilter, (1,10), padding="same", activation=tf.nn.tanh)

        self.layer2 = tf.keras.layers.Conv2D(self.nfilter, (1,10), strides=(1,2), padding="same", activation=tf.nn.tanh)
        self.layer2b = tf.keras.layers.Conv2D(self.nfilter, (1,10), strides=(1,1), padding="same", activation=tf.nn.tanh)

        self.layer3 = tf.keras.layers.Conv2D(self.nfilter, (1,6), strides=(1,2), padding="same", activation=tf.nn.tanh)
        self.layer3b = tf.keras.layers.Conv2D(self.nfilter, (1,6), strides=(1,1), padding="same", activation=tf.nn.tanh)

        self.layer4 = tf.keras.layers.Conv2D(self.nfilter, (1,3), strides=(1,2), padding="same", activation=tf.nn.tanh)
        self.layer4b = tf.keras.layers.Conv2D(self.nfilter, (1,3), strides=(1,1), padding="same", activation=tf.nn.tanh)

        self.layer5 = tf.keras.layers.Flatten()
        self.layer6 = tf.keras.layers.Dense(256, activation=tf.nn.tanh)
        self.layer6b = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs, training=False):
        tensor = tf.expand_dims(inputs, 1)
        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer2b(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer3b(tensor)
        tensor = self.layer4(tensor)
        tensor = self.layer4b(tensor)
        tensor = self.layer5(tensor)
        tensor = self.layer6(tensor)
        tensor = self.layer6b(tensor)

        return tensor


class Discriminator(tf.keras.Model):

    def __init__(self, maxdata, **kwargs):
        super().__init__(**kwargs)

        self.labelmerger = LabelMerger(maxdata)
        self.downstream = Downstream()
    
    @tf.function
    def call(self, inputs, training=False):
        tensor = self.labelmerger(inputs)
        tensor = self.downstream(tensor)

        return tensor


class WassersteinDistance(tf.keras.Model):

    def __init__(self, discriminator : Discriminator, **kwargs):
        super().__init__(**kwargs)

        self.discriminator = discriminator

    @tf.function
    def call(self, inputs, training=False):
        label = tf.cast(inputs[0], tf.float32)
        realdata = tf.cast(inputs[1], tf.float32)
        fakedata = tf.cast(inputs[2], tf.float32)

        realtensor = tf.reduce_mean(self.discriminator((label, realdata,)))
        faketensor = tf.reduce_mean(self.discriminator((label, fakedata,)))

        return (realtensor - faketensor)


class GradientPenalty(tf.keras.Model):

    def __init__(self, discriminator : Discriminator, **kwargs):
        super().__init__(**kwargs)

        self.discriminator = discriminator
        self.superposition = Superposition()

    @tf.function
    def call(self, inputs, training=False):
        label = tf.cast(inputs[0], tf.float32)
        realdata = tf.cast(inputs[1], tf.float32)
        fakedata = tf.cast(inputs[2], tf.float32)

        realmerged = self.discriminator.labelmerger((label, realdata,))
        fakemerged = self.discriminator.labelmerger((label, fakedata,))

        sup = self.superposition((realmerged, fakemerged,))
        tensor = self.discriminator.downstream(sup)

        gradnorm = tf.sqrt(tf.math.reduce_sum(tf.square(tf.gradients(tensor, sup)[0]), axis=[1,2]))
        gradient_penalty = tf.reduce_mean((gradnorm - 1)**2, axis=0)

        return gradient_penalty

