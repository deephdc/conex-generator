import src
import src.models.old.tfv2_generator as tfgen
import src.models.old.tfv2_discriminator as tfdis

import tensorflow as tf
import numpy as np
import timeit

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

label : tf.data.Dataset = data[2]
label = label.prefetch(int(2**16))

def noise_gen():
    for ii in range(5):
        yield np.random.uniform(-1,1,100*1024).reshape(1024,100)

noise = tf.data.Dataset.from_generator(noise_gen, tf.float32, (1024,100)).unbatch().repeat()

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

dopt = tf.keras.optimizers.Nadam()
gopt = tf.keras.optimizers.Nadam()

@tf.function
def train(generator, discriminator, wassersteindistance, gradientpenalty, dataset, optimizers):
    for x,y,z in dataset:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(discriminator.trainable_weights)
            fake = generator((x,y,))
            wd = wassersteindistance((x,z,fake,))
            gp = gradientpenalty((x,z,fake,))
            loss = - wd + 10*gp
        grads = tape.gradient(loss, discriminator.trainable_weights)
        optimizers.apply_gradients(zip(grads, discriminator.trainable_weights))
        break

train(generator, discriminator, wassersteindistance, gradientpenalty, ds, dopt)
