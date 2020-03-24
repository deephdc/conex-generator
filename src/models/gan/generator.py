import tensorflow as tf

from .utils.datamerger import DataSplitter
from .utils.datanormalizer import DataDenormalizer
from .utils.labelmerger import LabelMerger


class Generator(tf.keras.Model):

    def __init__(self, pd_maxdata, ed_maxdata, pd_feature_list=None,
                 ed_feature_list=None, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.pd_maxdata = pd_maxdata
        self.ed_maxdata = ed_maxdata
        self._numparticle = numparticle


        self.datasplitter = DataSplitter(pd_feature_list, ed_feature_list)
        self.denormalizer = DataDenormalizer(self.pd_maxdata, self.ed_maxdata)
        self.gen_features = self.datasplitter.gen_features

        self.dense_generator = DenseGenerator(self.gen_features,
                                              self._numparticle)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise1 = inputs[1]

        # run different generators
        output1 = self.dense_generator((label,noise1,))

        # merge outputs
        tensor = output1

        # format data
        tensor = self.datasplitter(tensor)
        tensor = self.denormalizer(tensor)

        return tensor


class DenseGenerator(tf.keras.Model):
    def __init__(self, gen_features=8, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.gen_features = gen_features

        self.labelmerger = LabelMerger(numparticle=numparticle)

        self.activation = tf.keras.activations.tanh

        self.layer1 = tf.keras.layers.Dense(512)
        self.layer2 = tf.keras.layers.Dense(1024)
        self.layer3 = tf.keras.layers.Dense(self.gen_features * 300)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise1 = inputs[1]

        labeltensor = self.labelmerger(label)
        tensor = tf.concat([labeltensor, noise1], -1)

        tensor = self.layer1(tensor)
        tensor = self.activation(tensor)

        tensor = self.layer2(tensor)
        tensor = self.activation(tensor)

        tensor = self.layer3(tensor)
        tensor = tf.keras.activations.sigmoid(tensor) * 1.5
        tensor = tf.reshape(tensor, shape=[-1, 300, self.gen_features])

        return tensor

