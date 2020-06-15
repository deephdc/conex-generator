import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import src
log = src.utils.getLogger(__name__, level="warning")

import tensorflow as tf
import numpy as np
import timeit
import typing
import json

import matplotlib
import matplotlib.pyplot as plt

# script input
run = "run03_test02"
cache_path = os.path.join("/home/tmp/koepke/cache", run)
save_prefix = "gan/epochs_3000/run03"
save_index = "02"

processbatchlen = 1000
labelbatchlen = 10
numfitdata = 100000

assert numfitdata % labelbatchlen == 0
assert processbatchlen >= labelbatchlen

# setup save env
fitdata_save_path = os.path.join(src.models.get_path(), save_prefix, save_index)
os.mkdir(fitdata_save_path)

# get data
data = src.data.processed.load_data(run)
pd : tf.data.Dataset = data[0]
ed : tf.data.Dataset = data[1]
label : tf.data.Dataset = data[2]
metadata : typing.Dict = data[3]

# parse metadata
numdata = metadata["length"]

pd_maxdata = np.array(metadata["particle_distribution"]["max_data"],
                      dtype=np.float32)
ed_maxdata = np.array(metadata["energy_deposit"]["max_data"],
                      dtype=np.float32)

pd_depth = metadata["particle_distribution"]["depth"]
pd_depthlen = len(pd_depth)
ed_depth = metadata["energy_deposit"]["depth"]
ed_depthlen = len(ed_depth)
assert pd_depthlen == ed_depthlen
depthlen = pd_depthlen

pd_mindepthlen = metadata["particle_distribution"]["min_depthlen"]
pd_mindepth = pd_depth[pd_mindepthlen-1]
ed_mindepthlen = metadata["energy_deposit"]["min_depthlen"]
ed_mindepth = ed_depth[ed_mindepthlen-1]
assert pd_mindepthlen == ed_mindepthlen
mindepthlen = pd_mindepthlen

# prepare data
def cast_to_float32(x):
    return tf.cast(x, tf.float32)

prefetchlen = int(3*1024*1024*1024 / (275 * 9 * 8)) # max 3 GB

pd = pd.prefetch(prefetchlen)
pd = pd.batch(prefetchlen//2) \
        .map(cast_to_float32,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .unbatch()
pd = src.data.cache_dataset(pd, "pd", basepath=cache_path)
pd = pd.prefetch(prefetchlen)

ed = ed.prefetch(prefetchlen)
ed = ed.batch(prefetchlen//2) \
        .map(cast_to_float32,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .unbatch()
ed = src.data.cache_dataset(ed, "ed", basepath=cache_path)
ed = ed.prefetch(prefetchlen)

label = label.prefetch(prefetchlen)
label = label.batch(prefetchlen//2) \
        .map(cast_to_float32,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .unbatch()
label = src.data.cache_dataset(label, "label", basepath=cache_path)
label = label.prefetch(prefetchlen)

noise1 = src.data.random.uniform_dataset((100,))

ds = tf.data.Dataset.zip((
    label,
    (pd, ed),
    (noise1,)
))
ds : tf.data.Dataset = ds.batch(processbatchlen).prefetch(2)

# load model
import src.models.gan as gan

genpath = os.path.join(src.models.get_path(), save_prefix, "generator")
dispath = os.path.join(src.models.get_path(), save_prefix, "discriminator")

gen = tf.keras.models.load_model(genpath)
dis = tf.keras.models.load_model(dispath)
wd = gan.loss.WassersteinDistance(dis)
gp = gan.loss.GradientPenalty(dis, pd_maxdata, ed_maxdata)

# initialize and build once
for label, real, noise in ds.take(1):
    out1 = gen([label, *noise,])
    out2 = dis([label, *real, *real,])
    out3 = wd([label, *real, *out1,])
    out4 = gp([label, *real, *out1,])

# create fitdata
gdata = np.zeros((numfitdata,depthlen,8), dtype=np.float32)
rdata = np.zeros((numfitdata,depthlen,8), dtype=np.float32)
flabel = np.zeros((numfitdata,5), dtype=np.float32)

index = 0
for label, real, noise in ds:
    if index >= numfitdata:
        break

    fake = gen([label, *noise])

    rpd = np.where(np.isnan(real[0]), 0.0, real[0])
    red = np.where(np.isnan(real[1]), 0.0, real[1])

    gpd = np.where(np.isnan(real[0]), 0.0, fake[0])
    ged = np.where(np.isnan(real[1]), 0.0, fake[1])

    batchlen = rpd.shape[0]

    ii = 0
    while ii < batchlen:
        curlabel = label[ii:ii+labelbatchlen]

        if len(curlabel) != labelbatchlen:
            break

        firstlabel = curlabel[0:1,0:-1]
        alllabel = curlabel[:,0:-1]
        same_label_set = np.all(alllabel == firstlabel)

        if not same_label_set:
            shiftindex = np.argmin(np.all(alllabel == firstlabel, axis=1))
            print("unaligned label set. shifting by", shiftindex)
            ii += shiftindex
            continue

        electron_or_gamma = curlabel[0,0] < 2
        if electron_or_gamma:
            ii += labelbatchlen
            continue

        rdata[index:index+labelbatchlen,:,0:-1] = rpd[ii:ii+labelbatchlen,:,0:-1]
        rdata[index:index+labelbatchlen,:,-1] =   red[ii:ii+labelbatchlen,:,-1]

        gdata[index:index+labelbatchlen,:,0:-1] = gpd[ii:ii+labelbatchlen,:,0:-1]
        gdata[index:index+labelbatchlen,:,-1] =   ged[ii:ii+labelbatchlen,:,-1]

        flabel[index:index+labelbatchlen,:] = curlabel[:,:]

        ii += labelbatchlen

        index += labelbatchlen
        if index >= numfitdata:
            break

print("processed data:", index)

# save data
rdata_path = os.path.join(fitdata_save_path, "rdata.npy")
gdata_path = os.path.join(fitdata_save_path, "gdata.npy")
label_path = os.path.join(fitdata_save_path, "label.npy")
metadata_path = os.path.join(fitdata_save_path, "metadata.json")

np.save(rdata_path, rdata, fix_imports=False)
np.save(gdata_path, gdata, fix_imports=False)
np.save(label_path, flabel, fix_imports=False)

with open(metadata_path, "w") as fp:
    json.dump(metadata, fp, indent=4)

