import tensorflow as tf

from .utils import DataSplitter
from .utils import DataDenormalizer
from .utils import LabelMerger


def get_maxdepthlen():
    return 2*2*2*2*2*3*3 # 288


class Generator(tf.keras.Model):

    def __init__(self, depthlen, pd_maxdata, ed_maxdata, pd_feature_list=None,
                 ed_feature_list=None, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.depthlen = depthlen
        self.maxdepthlen = get_maxdepthlen()
        self.pd_maxdata = pd_maxdata
        self.ed_maxdata = ed_maxdata
        self._numparticle = numparticle

        self.datasplitter = DataSplitter(pd_feature_list, ed_feature_list)
        self.denormalizer = DataDenormalizer(self.pd_maxdata, self.ed_maxdata)
        self.gen_features = self.datasplitter.gen_features

        self.dense_generator = DenseGenerator(self.maxdepthlen,
                                              self.gen_features,
                                              self._numparticle)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise = inputs[1]

        # run different generators
        output1 = self.dense_generator((label,noise,))

        # merge outputs
        tensor = output1

        # format data
        tensor = tensor[:,0:self.depthlen,:]
        tensor = self.datasplitter(tensor)
        tensor = self.denormalizer(tensor)

        return tensor


class DenseGenerator(tf.keras.Model):

    def __init__(self, depthlen=288, gen_features=8, numparticle=6, **kwargs):
        super().__init__(**kwargs)

        self.depthlen = depthlen
        self.gen_features = gen_features

        self.labelmerger = LabelMerger(numparticle=numparticle)

        self.activation = tf.keras.activations.tanh

        self.layer1 = tf.keras.layers.Dense(512)
        self.layer2 = tf.keras.layers.Dense(1024)
        self.layer3 = tf.keras.layers.Dense(self.depthlen * self.gen_features)

    @tf.function
    def call(self, inputs, training=False):
        label = inputs[0]
        noise = inputs[1][0]

        labeltensor = self.labelmerger(label)
        tensor = tf.concat([labeltensor, noise], -1)

        tensor = self.layer1(tensor)
        tensor = self.activation(tensor)

        tensor = self.layer2(tensor)
        tensor = self.activation(tensor)

        tensor = self.layer3(tensor)
        tensor = tf.keras.activations.sigmoid(tensor) * 1.5
        tensor = tf.reshape(tensor, shape=[-1, self.depthlen, self.gen_features])

        return tensor

