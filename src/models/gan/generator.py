import tensorflow as tf


class Generator(tf.keras.Model):

    def __init__(self, maxdata, numparticle = 6, **kwargs):
        super().__init__(**kwargs)
        self.maxdata = maxdata
        self._numparticle = numparticle

        self.first_dense = tf.keras.layers.Dense(512)

        self.dense_generator = DenseGenerator()


    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise1 = inputs[1]

        # normalize inputs
        particle = tf.cast(label[:,0], tf.int32)
        particle_oh = tf.cast(tf.one_hot(particle, self._numparticle), tf.float32)
        energy = label[:,1] / 1e10
        theta = label[:,2] / 90.0
        phi = label[:,3] / 180.0

        tensor = tf.concat(
            [
                noise,
                particle_oh,
                tf.expand_dims(energy, -1),
                tf.expand_dims(theta, -1),
                tf.expand_dims(phi, -1),
            ],
            -1
        )

        tensor = self.first_dense(tensor)

        output1 = self.dense_generator(tensor)

        tensor = output1

        temp0 = tf.expand_dims(tensor[:,:,0] * self.maxdata[0], -1)
        temp1 = tf.expand_dims(tensor[:,:,1] * self.maxdata[1], -1)
        temp2 = tf.expand_dims(tensor[:,:,2] * self.maxdata[2], -1)
        temp3 = tf.expand_dims(tensor[:,:,3] * self.maxdata[3], -1)
        temp4 = tf.expand_dims(tensor[:,:,4] * self.maxdata[4], -1)
        temp5 = tf.expand_dims(tensor[:,:,5] * self.maxdata[5], -1)
        temp6 = tf.expand_dims(tensor[:,:,6] * self.maxdata[6], -1)

        tensor = tf.concat(
                [temp0, temp1, temp2, temp3, temp4, temp5, temp6],
                -1)

        return tensor


class DenseGenerator(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.activation = tf.keras.activations.tanh

        self.layer1 = tf.keras.layers.Dense(512)
        self.layer2 = tf.keras.layers.Dense(4096)
        self.layer3 = tf.keras.layers.Dense(7*275)


    @tf.function
    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)

        tensor = self.activation(inputs)

        tensor = self.layer1(tensor)
        tensor = self.activation(tensor)

        tensor = self.layer2(tensor)
        tensor = self.activation(tensor)

        tensor = self.layer3(tensor)
        tensor = tf.keras.activations.sigmoid(tensor) * 1.5
        tensor = tf.reshape(tensor, shape=[input_shape[0], 275, 7])



        return tensor



noise = tf.random.uniform((300,100))
label = tf.ones((300,4))
maxdata = list(range(7))

gen = Generator(maxdata)
out = gen((noise,label,))

