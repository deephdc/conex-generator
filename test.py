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
batchsize = 512

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
        .add_summary_writer(savepath) \
        .build()

# execute training
print("started training")
training.execute("single", 10)
training.execute("random", 10)
training.execute("all", 10)
training.execute("random ensemble", 10)
training.execute("ensemble", 10)

exit(0)
# train function
start = timeit.default_timer()

end = timeit.default_timer()
print("training time", end-start)
exit(0)

# save model
print("saving ...")
gen.save(os.path.join(savepath, "generator"))
dis.save(os.path.join(savepath, "discriminator"))
print("done ...")

