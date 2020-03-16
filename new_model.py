import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import src
log = src.utils.getLogger(__name__, level="info")

import tensorflow as tf
import numpy as np
import timeit
import typing

# get data
data = src.data.processed.load_data("run03")
pd : tf.data.Dataset = data[0]
ed : tf.data.Dataset = data[1]
label : tf.data.Dataset = data[2]
metadata : typing.Dict = data[3]

# parse metadata
numdata = metadata["length"]

pd_depth = metadata["particle_distribution"]["depth"]
pd_depthlen = len(pd_depth)

ed_depth = metadata["energy_deposit"]["depth"]
ed_depthlen = len(ed_depth)

prefetchlen = int(3*1024*1024*1024 / (275 * 9 * 8)) # max 3 GB

# prepare data
def cast_to_float32(x):
    return tf.cast(x, tf.float32)

pd = pd.prefetch(prefetchlen)
pd = pd.batch(prefetchlen//2) \
        .map(cast_to_float32,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .unbatch()
pd = src.data.cache_dataset(pd, "pd")
pd = pd.prefetch(prefetchlen)

ed = ed.prefetch(prefetchlen)
ed = ed.batch(prefetchlen//2) \
        .map(cast_to_float32,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .unbatch()
ed = src.data.cache_dataset(ed, "ed")
ed = ed.prefetch(prefetchlen)

label = label.prefetch(prefetchlen)
label = label.batch(prefetchlen//2) \
        .map(cast_to_float32,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .unbatch()
label = src.data.cache_dataset(label, "label")
label = label.prefetch(prefetchlen)


noise = src.data.random.uniform_dataset((100,))

ds = tf.data.Dataset.zip((label, noise, pd))
ds = ds.shuffle(int(2**14)).batch(1024).prefetch(4)

exit()

maxdata = [0,0,0,0,0,0,0,0]
for x in ds:
    curmax = tf.math.reduce_max(x[2], axis=(0,1))
    for ii, cm in enumerate(maxdata):
        if cm < curmax[ii]:
            maxdata[ii] = curmax[ii]

