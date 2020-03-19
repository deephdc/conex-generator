import tensorflow as tf


class DataNormalizer(tf.keras.layers.Layer):

    def __init__(self, pd_maxdata, ed_maxdata, **kwargs):
        super().__init__(**kwargs)

        self._pd_maxdata = tf.constant(pd_maxdata, dtype=tf.float32)
        self._ed_maxdata = tf.constant(ed_maxdata, dtype=tf.float32)

    @tf.function
    def call(self, inputs, training=False):
        pd = inputs[0]
        ed = inputs[1]

        pd_maxdata = tf.where(self._pd_maxdata == 0.0, 1.0, self._pd_maxdata)
        ed_maxdata = tf.where(self._ed_maxdata == 0.0, 1.0, self._ed_maxdata)

        pd = pd / pd_maxdata
        ed = ed / ed_maxdata

        return (pd, ed)


class DataDenormalizer(tf.keras.layers.Layer):

    def __init__(self, pd_maxdata, ed_maxdata, **kwargs):
        super().__init__(**kwargs)

        self._pd_maxdata = tf.constant(pd_maxdata, dtype=tf.float32)
        self._ed_maxdata = tf.constant(ed_maxdata, dtype=tf.float32)

    @tf.function
    def call(self, inputs, training=False):
        pd = inputs[0]
        ed = inputs[1]

        pd_maxdata = tf.where(self._pd_maxdata == 0.0, 1.0, self._pd_maxdata)
        ed_maxdata = tf.where(self._ed_maxdata == 0.0, 1.0, self._ed_maxdata)

        pd = pd * pd_maxdata
        ed = ed * ed_maxdata

        return (pd, ed)

