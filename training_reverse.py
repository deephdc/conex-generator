import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import src
log = src.utils.getLogger(__name__, level="info")

import tensorflow as tf
import numpy as np
import timeit

# script input
run = "run03"
save_prefix = "gan/reverse"
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

# slowly increase learning rate
training.learning_rate(1e-6, 1e-6).execute("all", 1000)
training.learning_rate(2e-6, 2e-6).execute("all", 1000)
training.learning_rate(4e-6, 4e-6).execute("all", 1000)
training.learning_rate(1e-5, 1e-5).execute("all", 1000)
training.learning_rate(2e-5, 2e-5).execute("all", 1000)
training.learning_rate(4e-5, 4e-5).execute("all", 1000)

# decorrelate submodels
training.learning_rate(1e-4, 1e-4).execute("all", 10000)

# random correlate submodels
training.learning_rate(1e-5, 1e-5).execute("random", 5000)
training.learning_rate(1e-4, 1e-4).execute("random", 5000)

# correlate submodels
training.learning_rate(1e-4, 1e-4).execute("single", 20000)

# random weight submodels
training.learning_rate(1e-5, 1e-5).execute("random ensemble", 5000)
training.learning_rate(1e-4, 1e-4).execute("random ensemble", 5000)

# weight submodels
training.learning_rate(1e-4, 1e-4).execute("ensemble", 10000)

# slowly decrease learning rate
training.learning_rate(4e-5, 4e-5).execute("ensemble", 10000)
training.learning_rate(2e-5, 2e-5).execute("ensemble", 10000)
training.learning_rate(1e-5, 1e-5).execute("ensemble", 10000)
training.learning_rate(4e-6, 4e-6).execute("ensemble", 10000)
training.learning_rate(2e-6, 2e-6).execute("ensemble", 10000)
training.learning_rate(1e-6, 1e-6).execute("ensemble", 10000)

end = timeit.default_timer()
print("training time", end-start)

# save model
print("saving ...")
model.save(savepath)
print("done ...")

