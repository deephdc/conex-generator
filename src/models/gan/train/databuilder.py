import tensorflow as tf
import numpy as np
import os

import src.utils
import src.data


class DataBuilder():

    def __init__(self, run):
        self.run = run

        self.dataset : tf.data.Dataset = None

        data = src.data.processed.load_data(self.run)
        self.pd : tf.data.Dataset = data[0]
        self.ed : tf.data.Dataset = data[1]
        self.label : tf.data.Dataset = data[2]
        self.metadata : typing.Dict = data[3]

        self.parse_metadata(self.metadata)

    def parse_metadata(self, metadata):
        self.numdata = metadata["length"]

        self.pd_maxdata = np.array(
                metadata["particle_distribution"]["max_data"],
                dtype=np.float32)

        self.ed_maxdata = np.array(
                metadata["energy_deposit"]["max_data"],
                dtype=np.float32)

        self.pd_depth = metadata["particle_distribution"]["depth"]
        pd_depthlen = len(self.pd_depth)
        self.ed_depth = metadata["energy_deposit"]["depth"]
        ed_depthlen = len(self.ed_depth)
        assert pd_depthlen == ed_depthlen
        self.depthlen = pd_depthlen

        pd_mindepthlen = metadata["particle_distribution"]["min_depthlen"]
        self.pd_mindepth = self.pd_depth[pd_mindepthlen-1]
        ed_mindepthlen = metadata["energy_deposit"]["min_depthlen"]
        self.ed_mindepth = self.ed_depth[ed_mindepthlen-1]
        assert pd_mindepthlen == ed_mindepthlen
        self.mindepthlen = pd_mindepthlen

        return self

    def build(self):
        noise1 = src.data.random.uniform_dataset((100,))

        self.dataset : tf.data.Dataset = tf.data.Dataset.zip((
            self.label,
            (self.pd, self.ed),
            (noise1,)
        ))

        return self

    def prefetch(self, value):
        if isinstance(value, str):
            fetch_bytes = src.utils.parse_byte_string(value)

            datum = [x for x in self.ed.take(1)][0]
            shape = datum.shape
            dtype = datum.dtype
            bytes_per_datum = np.product([*shape]) * dtype.size

            value = np.ceil(fetch_bytes / bytes_per_datum)

        self.pd : tf.data.Dataset = self.pd.prefetch(int(value))
        self.ed : tf.data.Dataset = self.ed.prefetch(int(value))
        self.label : tf.data.Dataset = self.label.prefetch(int(value))

        return self

    def batch(self, value):
        if isinstance(value, str):
            fetch_bytes = src.utils.parse_byte_string(value)

            datum = [x for x in self.ed.take(1)][0]
            shape = datum.shape
            dtype = datum.dtype
            bytes_per_datum = np.product([*shape]) * dtype.size

            value = np.ceil(fetch_bytes / bytes_per_datum)

        self.pd : tf.data.Dataset = self.pd.batch(int(value))
        self.ed : tf.data.Dataset = self.ed.batch(int(value))
        self.label : tf.data.Dataset = self.label.batch(int(value))

        return self

    def unbatch(self):
        self.pd : tf.data.Dataset = self.pd.unbatch()
        self.ed : tf.data.Dataset = self.ed.unbatch()
        self.label : tf.data.Dataset = self.label.unbatch()

        return self

    def map_pd(self, fn):
        self.pd : tf.data.Dataset = self.pd.map(
                fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def map_ed(self, fn):
        self.ed : tf.data.Dataset = self.ed.map(
                fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def map_label(self, fn):
        self.label : tf.data.Dataset = self.label.map(
                fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self

    def cast_to_float32(self):
        def map_fn(x):
            return tf.cast(x, tf.float32)

        self.batch(1024*1024)

        self.map_pd(map_fn)
        self.map_ed(map_fn)
        self.map_label(map_fn)

        self.unbatch()

        return self

    def cache(self, base_path="/home/tmp/koepke/cache", **kwargs):
        cache_path = os.path.join(base_path, self.run)

        self.pd : tf.data.Dataset = src.data.cache_dataset(
                self.pd, "pd", basepath=cache_path, **kwargs)
        self.ed : tf.data.Dataset = src.data.cache_dataset(
                self.ed, "ed", basepath=cache_path, **kwargs)
        self.label : tf.data.Dataset = src.data.cache_dataset(
                self.label, "label", basepath=cache_path, **kwargs)
        
        return self

