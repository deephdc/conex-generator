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

noise1 = src.data.random.uniform_dataset((100,))

ds = tf.data.Dataset.zip((
    label,
    (pd, ed),
    (noise1,)
))
ds : tf.data.Dataset = ds.shuffle(100000).batch(1024).prefetch(5)

# get data info: maximum estimate, shape
@tf.function
def get_maxdata(dataset):
    pd_maxdata = tf.zeros(8, dtype=tf.float32)
    ed_maxdata = tf.zeros(9, dtype=tf.float32)
    for batch in dataset:
        # pd
        batchmax = tf.math.reduce_max(tf.abs(batch[1][0]), axis=(0,1))
        pd_maxdata = tf.where(batchmax > pd_maxdata, batchmax, pd_maxdata)

        # ed
        batchmax = tf.math.reduce_max(tf.abs(batch[1][1]), axis=(0,1))
        ed_maxdata = tf.where(batchmax > ed_maxdata, batchmax, ed_maxdata)

    return (pd_maxdata, ed_maxdata)
pd_maxdata, ed_maxdata = get_maxdata(ds)

pd_maxdata = pd_maxdata.numpy()
ed_maxdata = ed_maxdata.numpy()

for batch in ds.take(1):
    depthlen  = batch[1][0].shape[1]

# tests
import src.models.gan as gan

gen = gan.BaseGenerator(depthlen, pd_maxdata, ed_maxdata)
dis = gan.BaseDiscriminator(pd_maxdata, ed_maxdata)
wd = gan.loss.WassersteinDistance(dis)
gp = gan.loss.GradientPenalty(dis)

@tf.function
def testfunc(dataset, gen, dis, wd, gp):
    for label,real,noise in dataset:
        fake = gen((label,noise,))
        out1 = wd((label,real,fake))
        out2 = gp((label,real,fake))

start = timeit.default_timer()
retval = testfunc(ds, gen, dis, wd, gp)
end = timeit.default_timer()

