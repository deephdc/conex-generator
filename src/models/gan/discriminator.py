import tensorflow as tf


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

