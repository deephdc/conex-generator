import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import src
log = src.utils.getLogger(__name__, level="warning")

import tensorflow as tf
import numpy as np
import timeit
import typing

# get data
data = src.data.processed.load_data("run02")
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

ds = tf.data.Dataset.zip((label, pd, ed))
ds : tf.data.Dataset = ds.shuffle(100000).batch(1024).prefetch(5)

noise = src.data.random.uniform_dataset((100,))

# get data maximum estimate
pd_maxdata = tf.zeros(8, dtype=tf.float32)
ed_maxdata = tf.zeros(9, dtype=tf.float32)
for batch in ds.take(100):
    # pd
    batchmax = tf.math.reduce_max(tf.abs(batch[1]), axis=(0,1))
    pd_maxdata = tf.where(batchmax > pd_maxdata, batchmax, pd_maxdata)

    # ed
    batchmax = tf.math.reduce_max(tf.abs(batch[2]), axis=(0,1))
    ed_maxdata = tf.where(batchmax > ed_maxdata, batchmax, ed_maxdata)

# tests
import src.models.gan as gan

dn = gan.utils.DataNormalizer(pd_maxdata, ed_maxdata)
dd = gan.utils.DataDenormalizer(pd_maxdata, ed_maxdata)
lm = gan.utils.LabelMerger()
dmerger = gan.utils.DataMerger()
dsplitter = gan.utils.DataSplitter()

for x in ds.take(1):
    pdd = tf.where(tf.math.is_nan(x[1]), 0.0, x[1])
    edd = tf.where(tf.math.is_nan(x[2]), 0.0, x[2])
    
    r1,r2 = dn((pdd, edd))
    s1,s2 = dd((r1,r2))

    a = dmerger((pdd, edd))
    t1,t2 = dsplitter(a)

d1 = tf.abs(pdd - s1)
d2 = tf.abs(edd - s2)

