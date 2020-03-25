import tensorflow as tf
import numpy as np


def get_default_pd_feature_list():
    default = [0,1,2,3,4,5,6]

    retval = list(set(default))
    retval.sort()
    return retval

def get_default_ed_feature_list():
    default = [8]

    retval = list(set(default))
    retval.sort()
    return retval


class DataMerger(tf.keras.layers.Layer):

    def __init__(self, pd_feature_list=None, ed_feature_list=None, **kwargs):
        super().__init__(**kwargs)

        if pd_feature_list is None:
            self._pd_feature_list = get_default_pd_feature_list()
        else:
            self._pd_feature_list = pd_feature_list

        if ed_feature_list is None:
            self._ed_feature_list = get_default_ed_feature_list()
        else:
            self._ed_feature_list = ed_feature_list

        self.gen_features = len(self._pd_feature_list) \
                            + len(self._ed_feature_list)

    @tf.function
    def call(self, inputs, training=False):
        particle_distribution = inputs[0]
        energy_deposit = inputs[1]

        pd_part = tf.gather(particle_distribution, self._pd_feature_list,
                            axis=2)
        ed_part = tf.gather(energy_deposit, self._ed_feature_list,
                            axis=2)
        tensor = tf.concat([pd_part, ed_part], -1)

        return tensor


class DataSplitter(tf.keras.layers.Layer):

    def __init__(self, pd_feature_list=None, ed_feature_list=None, **kwargs):
        super().__init__(**kwargs)

        if pd_feature_list is None:
            self._pd_feature_list = get_default_pd_feature_list()
        else:
            self._pd_feature_list = pd_feature_list

        if ed_feature_list is None:
            self._ed_feature_list = get_default_ed_feature_list()
        else:
            self._ed_feature_list = ed_feature_list

        self._pd_gatherindex = self.get_gatherindex(self._pd_feature_list, 8)
        self._ed_gatherindex = self.get_gatherindex(self._ed_feature_list, 9)

        self.gen_features = len(self._pd_feature_list) \
                            + len(self._ed_feature_list)

    @staticmethod
    def get_gatherindex(feature_list, numfeatures):
        gatherindex = np.arange(numfeatures) + len(feature_list)
        for ii,_ in enumerate(gatherindex):
            if ii in feature_list:
                gatherindex[ii] = feature_list.index(ii)
        return gatherindex

    def build(self, input_shape):
        self._depthlen = input_shape[1]

        pd_numfeatures = len(self._pd_feature_list)
        ed_numfeatures = len(self._ed_feature_list)
        assert pd_numfeatures + ed_numfeatures == input_shape[2]

        self._pd_dummy = tf.cast(
                tf.fill([1, self._depthlen, 8], np.nan),
                tf.float32)
        self._ed_dummy = tf.cast(
                tf.fill([1, self._depthlen, 9], np.nan),
                tf.float32)

        super().build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        splitindex = len(self._pd_feature_list)
        pd_part = inputs[:,:,:splitindex]
        ed_part = inputs[:,:,splitindex:]

        current_batchsize = tf.shape(inputs)[0]
        pd_dummy = tf.tile(self._pd_dummy, [current_batchsize, 1, 1])
        ed_dummy = tf.tile(self._ed_dummy, [current_batchsize, 1, 1])

        pd_temp = tf.concat([pd_part, pd_dummy], -1)
        ed_temp = tf.concat([ed_part, ed_dummy], -1)

        pd = tf.gather(pd_temp, self._pd_gatherindex, axis=2)
        ed = tf.gather(ed_temp, self._ed_gatherindex, axis=2)

        return (pd, ed)

