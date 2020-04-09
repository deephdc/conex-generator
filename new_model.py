import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import src
log = src.utils.getLogger(__name__, level="warning")

import tensorflow as tf
import numpy as np
import timeit
import typing

# script input
run = "run02"
cache_path = os.path.join("/home/tmp/koepke/cache", run)
epochs = 4000

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
ds : tf.data.Dataset = ds.shuffle(100000).batch(1024).prefetch(5)

# create model
import src.models.gan as gan

gen = gan.BaseGenerator(depthlen, pd_maxdata, ed_maxdata)
dis = gan.BaseDiscriminator(pd_maxdata, ed_maxdata)
wd = gan.loss.WassersteinDistance(dis)
gp = gan.loss.GradientPenalty(dis)

gopt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
dopt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

# initialize and build once
for label, real, noise in ds.take(1):
    out1 = gen([label, *noise,])
    out2 = dis([label, *real, *real,])
    out3 = wd([label, *real, *out1,])
    out4 = gp([label, *real, *out1,])

# train function
def train(dataset, gen, dis, wd, gp, gopt, dopt, epochs):
    dataset = dataset.repeat(epochs)
    for ii, (label, real, noise) in dataset.enumerate():
        if ii % 5 == 4:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(gen.trainable_weights)
                fake = gen([label, *noise,])
                distance = wd([label, *real, *fake,])
                loss = distance
            grads = tape.gradient(loss, gen.trainable_weights)
            dopt.apply_gradients(zip(grads, gen.trainable_weights))
        else:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(dis.trainable_weights)
                fake = gen([label, *noise,])
                distance = wd([label, *real, *fake,])
                penalty = gp([label, *real, *fake,])
                loss = - distance + penalty
            grads = tape.gradient(loss, dis.trainable_weights)
            dopt.apply_gradients(zip(grads, dis.trainable_weights))

print("training ...")
start = timeit.default_timer()
train(ds, gen, dis, wd, gp, gopt, dopt, epochs)
end = timeit.default_timer()
print("training time", end-start)

# save model
print("saving ...")
savepath = os.path.join(src.models.get_path(), "gan", run)

for label, real, noise in ds.take(1):
    gen.predict([label, *noise,])
    dis.predict([label, *real, *real,])

gen.save(os.path.join(savepath, "generator"))
dis.save(os.path.join(savepath, "discriminator"))
print("done ...")

