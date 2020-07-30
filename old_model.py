"""General script to train the old TensorFlow 1 model in /src/models/old"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
import timeit

import src
import src.models.old.tfv2_generator as tfgen
import src.models.old.tfv2_discriminator as tfdis

data = src.data.processed.load_data("run01")

def nantozero(tensor):
    return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

def extrazero(tensor):
    shape = tf.shape(tensor)
    zeros = tf.zeros((shape[0], 256-shape[1], shape[2],), dtype=tensor.dtype)
    return tf.concat([tensor, zeros,], axis=1)

pd : tf.data.Dataset = data[0]
pd = pd.prefetch(int(2**16))
pd = pd.batch(1024).map(nantozero, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
pd = pd.batch(1024).map(extrazero, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()

pd = pd.cache("/home/tmp/pd")
start = timeit.default_timer()
for x in pd:
    pass
end = timeit.default_timer()
print("cache time", end-start)

label : tf.data.Dataset = data[2]
label = label.prefetch(int(2**16))

label = label.cache("/home/tmp/label")
start = timeit.default_timer()
for x in label:
    pass
end = timeit.default_timer()
print("cache time", end-start)

def noise_gen():
    for ii in range(5):
        yield np.random.uniform(-1,1,100*1024*1024).reshape(1024*1024,100)

noise = tf.data.Dataset.from_generator(noise_gen, tf.float32, (1024*1024,100)).unbatch().repeat()

ds = tf.data.Dataset.zip((label, noise, pd))
ds = ds.shuffle(int(2**14)).batch(1024).prefetch(4)

maxdata = [0,0,0,0,0,0,0,0]
for x in ds:
    curmax = tf.math.reduce_max(x[2], axis=(0,1))
    for ii, cm in enumerate(maxdata):
        if cm < curmax[ii]:
            maxdata[ii] = curmax[ii]


generator = tfgen.Generator(maxdata)
discriminator = tfdis.Discriminator(maxdata)
wassersteindistance = tfdis.WassersteinDistance(discriminator)
gradientpenalty = tfdis.GradientPenalty(discriminator)

for x,y,z in ds:
    fake = generator((x,y,))
    wd = wassersteindistance((x,z,fake,))
    break

dopt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
gopt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

def train(gen, dis, wd, gp, dataset : tf.data.Dataset, dopt, gopt, epochs):
    ds = dataset.repeat(epochs)
    try:
        for ii, (label,noise,data) in iter(ds.enumerate()):
            if ii % 5 == 4:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(gen.trainable_weights)
                    fake = gen((label, noise,))
                    distance = wd((label, data, fake,))
                    loss = distance
                grads = tape.gradient(loss, gen.trainable_weights)
                dopt.apply_gradients(zip(grads, gen.trainable_weights))
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(dis.trainable_weights)
                    fake = gen((label, noise,))
                    distance = wd((label, data, fake,))
                    penalty = gp((label, data, fake,))
                    loss = - distance + 10 * penalty
                grads = tape.gradient(loss, dis.trainable_weights)
                dopt.apply_gradients(zip(grads, dis.trainable_weights))
    except KeyboardInterrupt:
        pass

print("training ...")
start = timeit.default_timer()
train(generator, discriminator, wassersteindistance, gradientpenalty, ds, dopt, gopt, 15000)
end = timeit.default_timer()
print("training time", end-start)

for x,y,z in ds.take(1):
    test = generator.predict((x,y,))
generator.save("./output/old")
print("done ...")

