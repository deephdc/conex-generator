import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import src
log = src.utils.getLogger(__name__, level="warning")

import tensorflow as tf
import numpy as np
import timeit
import typing

# script input
run = "run01"
save_prefix = "test"
cache_path = "/home/tmp/koepke/cache"
epochs = 1
batchsize = 128

# input processing
savepath = os.path.join(src.models.get_path(), save_prefix, run)
logpath = os.path.join(savepath, "log")

# get data
import src.models.gan as gan

data = gan.train.DataBuilder(run, batchsize) \
        .prefetch("1 GB") \
        .cast_to_float32() \
        .cache(cache_path) \
        .build()

ds : tf.data.Dataset = data.dataset

# create model
model = gan.train.ModelBuilder(data) \
        .build()

# prepare training
training = gan.train.TrainBuilder(data, model) \
        .learning_rate(0.0001, 0.0001) \
        .summary_writer(savepath) \
        .build()

# execute training
training.execute("first", 1)

# train function
exit(0)
def train(dataset, gen, dis, wd, gp, gopt, dopt, epochs):
    dataset = dataset.repeat(epochs)
    for ii, (label, real, noise) in dataset.enumerate():
        if ii % 5 == 4:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(gen.trainable_weights)
                fake = gen([label, *noise,])
                distance = wd([label, *real, *fake,])
                loss = distance

            tf.summary.scalar("Wasserstein Distance", distance, step=ii)

            grads = tape.gradient(loss, gen.trainable_weights)
            gopt.apply_gradients(zip(grads, gen.trainable_weights))

        else:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(dis.trainable_weights)
                fake = gen([label, *noise,])
                distance = wd([label, *real, *fake,])
                penalty = gp([label, *real, *fake,])
                loss = - distance + penalty

            tf.summary.scalar("Wasserstein Distance", distance, step=ii)
            tf.summary.scalar("Gradient Penalty", penalty, step=ii)
            tf.summary.scalar("Discriminator Loss", loss, step=ii)

            grads = tape.gradient(loss, dis.trainable_weights)
            dopt.apply_gradients(zip(grads, dis.trainable_weights))

print("training ...")
start = timeit.default_timer()
with writer.as_default():
    try:
        train(ds, gen, dis, wd, gp, gopt, dopt, epochs)
    except KeyboardInterrupt:
        pass
    finally:
        writer.flush()
end = timeit.default_timer()
print("training time", end-start)

# save model
print("saving ...")
gen.save(os.path.join(savepath, "generator"))
dis.save(os.path.join(savepath, "discriminator"))
print("done ...")

