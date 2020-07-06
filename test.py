import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import src
log = src.utils.getLogger(__name__, level="info")

import tensorflow as tf
import numpy as np
import timeit
import typing

# script input
run = "run01"
save_prefix = "test"
cache_path = "/home/tmp/koepke/cache"
batchsize = 512

# input processing
savepath = os.path.join(src.models.get_path(), save_prefix, run)

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
        .learning_rate(1e-4, 1e-4) \
        .add_summary_writer(savepath) \
        .build()

# execute training
print("started training")
start = timeit.default_timer()
training.execute("single", 10)
training.execute("random", 10)
training.execute("all", 10)
training.execute("random ensemble", 10)
training.execute("ensemble", 10)
end = timeit.default_timer()
print("training time", end-start)

# save model
print("saving ...")
model.save(savepath)
print("done ...")

