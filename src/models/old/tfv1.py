import sys
print(sys.version)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
from datetime import datetime
import timeit
import time

import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plot

import logging
logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s - %(levelname)-8.8s: %(name)s: %(message)s",
        handlers = [
            logging.FileHandler("log.txt"),
            logging.StreamHandler(),
            ]
        )
log = logging.getLogger(__name__)

# =========
# load data
# =========
datadir = "/cr/users/koepke/data/conex"
data : np.ndarray = np.load(os.path.join(datadir, "data.npy"))[:,:,0:8]
label : np.ndarray = np.load(os.path.join(datadir, "label.npy"))

testlen = 10000

# normalization constants
maxdata = np.zeros(data.shape[2])
for ii in range(len(maxdata)):
    maxdata[ii] = np.max(data[:,:,ii])

maxenergy = 1.0e10
maxtheta = 65.0
maxphi = 180.0

# rescale depthlen to power of two
depthlen = data.shape[1]
addzeros = np.zeros((data.shape[0], 256, data.shape[2]))
addzeros[:,0:depthlen,:] = data
data = addzeros
gc.collect()

log.info("data loaded")


# ==============
# data formating
# ==============
depthlen = data.shape[1]
numchannels = data.shape[2]
numclasses = np.max(label[:,0]) + 1
noisesize = 100

xin = tf.placeholder(data.dtype, shape = [None, depthlen, numchannels])
yin = tf.placeholder(label.dtype, shape = [None, label.shape[1]])
nin = tf.placeholder(tf.float32, shape = [None, noisesize])
trainflag_generator = tf.placeholder(tf.bool, shape = [])
trainflag_discriminator = tf.placeholder(tf.bool, shape = [])

x_true = tf.cast(xin, tf.float32)

y_classes = tf.cast(yin[:,0], tf.int32)
y_classes_onehot = tf.cast(tf.one_hot(y_classes, numclasses), tf.float32)
y_energy = tf.cast(yin[:,1], tf.float32)
y_theta = tf.cast(yin[:,2], tf.float32)
y_phi = tf.cast(yin[:,3], tf.float32)


# ================
# helper functions
# ================
def yieldbatch(x : np.ndarray, y : np.ndarray, batchsize : int) -> (np.ndarray, np.ndarray, int, int):
    assert len(x) == len(y)

    length = len(x)
    steps = list(range(0, length, batchsize))
    splitindex = zip(steps, steps[1:] + [length])
    numsteps = len(steps)

    perm = np.random.permutation(length)
    tx = x[perm]
    ty = y[perm]
    tn = np.random.uniform(-1.0, 1.0, [length, noisesize])

    for curstep, index in enumerate(splitindex):
        yield (
                tx[index[0]:index[1]],
                ty[index[0]:index[1]],
                tn[index[0]:index[1]],
                curstep+1,
                numsteps,
        )

def normalize_data(tensor, maxdata):
    tensorparts = []
    for ii in range(len(maxdata)):
        tensorparts.append(tf.expand_dims(tensor[:,:,ii] / maxdata[ii], -1))
    return tf.concat(tensorparts, -1)

def denormalize_data(tensor, maxdata):
    tensorparts = []
    for ii in range(len(maxdata)):
        tensorparts.append(tf.expand_dims(tensor[:,:,ii] * maxdata[ii], -1))
    return tf.concat(tensorparts, -1)


# ==============
# neural network
# ==============
generatorvariables = []
discriminatorvariables = []


# generator
variables = []

tensor = tf.concat(
    [
        nin,
        y_classes_onehot,
        tf.expand_dims(y_energy, -1) / maxenergy,
        tf.expand_dims(y_theta, -1) / maxtheta,
        tf.expand_dims(y_phi, -1) / maxphi,
    ],
    -1
)

nheight = 4
nwidth = 16
nfilter = 64

layer = tf.layers.Dense(512, activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Dense(1024, activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Dense(2048, activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Dense(4096, activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Dense(nheight*nwidth*nfilter, activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

tensor = tf.reshape(tensor, [tf.shape(tensor)[0], nheight, nwidth, nfilter])

# 1st transpose the conv
layer = tf.keras.layers.Conv2DTranspose(nfilter, (2,5), strides=(1,2), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Conv2D(nfilter//2, (2,5), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

# 1st additional
layer = tf.keras.layers.Conv2DTranspose(nfilter//2, (2,10), strides=(1,1), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Conv2D(nfilter//2, (2,10), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

# 2nd transpose the conv
layer = tf.keras.layers.Conv2DTranspose(nfilter//2, (2,10), strides=(1,4), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Conv2D(nfilter//4, (2,10), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

# 2nd additional
layer = tf.keras.layers.Conv2DTranspose(nfilter//4, (2,5), strides=(1,1), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

layer = tf.layers.Conv2D(nfilter//4, (2,5), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

# 3rd transpose the conv
layer = tf.keras.layers.Conv2DTranspose(nfilter//4, (2,5), strides=(1,2), padding="same", activation=tf.nn.tanh)
tensor = layer(tensor)
variables += layer.trainable_weights

tensor = tf.pad(tensor, ([0,0], [0,0], [2,2], [0,0]))
layer = tf.layers.Conv2D(numchannels, (4,5), padding="valid", activation=tf.nn.sigmoid)
tensor = layer(tensor)
variables += layer.trainable_weights

# remove dimensions of size 1. shape should be [None, 256, 8]
tensor = tf.squeeze(tensor, [1])

generatoroutput = denormalize_data(tensor, maxdata)
generatorvariables = variables


# discriminator
variables = []

labeltensor = tf.concat(
    [
        y_classes_onehot,
        tf.expand_dims(y_energy, -1) / maxenergy,
        tf.expand_dims(y_theta, -1) / maxtheta,
        tf.expand_dims(y_phi, -1) / maxphi,
    ],
    -1
)

layer = tf.layers.Dense(4096, activation=tf.nn.tanh)
labeltensor = layer(labeltensor)
variables += layer.trainable_weights

layer = tf.layers.Dense(8192, activation=tf.nn.tanh)
labeltensor = layer(labeltensor)
variables += layer.trainable_weights

layer = tf.layers.Dense(depthlen*numchannels, activation=tf.nn.tanh)
labeltensor = tf.reshape(layer(labeltensor), [tf.shape(labeltensor)[0], depthlen, numchannels])
variables += layer.trainable_weights

discriminator_realinput = tf.concat(
        [
            normalize_data(x_true, maxdata),
            labeltensor,
        ],
        -1
)

discriminator_fakeinput = tf.concat(
        [
            normalize_data(generatoroutput, maxdata),
            labeltensor,
        ],
        -1
)

epsilon = tf.random.uniform(shape = [tf.shape(x_true)[0], 1, 1], dtype=tf.float32)
discriminator_gradinput = epsilon * discriminator_realinput + (1 - epsilon) * discriminator_fakeinput

layer1 = tf.layers.Conv2D(nfilter, (1,10), padding="same", activation=tf.nn.tanh)

layer2 = tf.layers.Conv2D(nfilter, (1,10), strides=(1,2), padding="same", activation=tf.nn.tanh)
layer2b = tf.layers.Conv2D(nfilter, (1,10), strides=(1,1), padding="same", activation=tf.nn.tanh)

layer3 = tf.layers.Conv2D(nfilter, (1,6), strides=(1,2), padding="same", activation=tf.nn.tanh)
layer3b = tf.layers.Conv2D(nfilter, (1,6), strides=(1,1), padding="same", activation=tf.nn.tanh)

layer4 = tf.layers.Conv2D(nfilter, (1,3), strides=(1,2), padding="same", activation=tf.nn.tanh)
layer4b = tf.layers.Conv2D(nfilter, (1,3), strides=(1,1), padding="same", activation=tf.nn.tanh)

layer5 = tf.layers.Flatten()
layer6 = tf.layers.Dense(256, activation=tf.nn.tanh)
layer6b = tf.layers.Dense(1)

def discriminator_map(tensor):
    tensor = tf.expand_dims(tensor, 1)
    tensor = layer1(tensor)
    tensor = layer2(tensor)
    tensor = layer2b(tensor)
    tensor = layer3(tensor)
    tensor = layer3b(tensor)
    tensor = layer4(tensor)
    tensor = layer4b(tensor)
    tensor = layer5(tensor)
    tensor = layer6(tensor)
    tensor = layer6b(tensor)

    return tensor

discriminator_reallogits = discriminator_map(discriminator_realinput)
discriminator_fakelogits = discriminator_map(discriminator_fakeinput)
discriminator_gradlogits = discriminator_map(discriminator_gradinput)

variables += layer1.trainable_weights
variables += layer2.trainable_weights
variables += layer2b.trainable_weights
variables += layer3.trainable_weights
variables += layer3b.trainable_weights
variables += layer4.trainable_weights
variables += layer4b.trainable_weights
variables += layer6.trainable_weights
variables += layer6b.trainable_weights
discriminatorvariables = variables

# loss and optimization
wasserstein_distance = tf.reduce_mean(discriminator_reallogits) - tf.reduce_mean(discriminator_fakelogits)
summary_wasserstein_distance = tf.summary.scalar("wasserstein_distance_discriminator", wasserstein_distance)

cross_entropy_discriminator = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.ones_like(discriminator_reallogits),
            logits = discriminator_reallogits) +
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels = tf.zeros_like(discriminator_fakelogits),
            logits = discriminator_fakelogits)
        )

cross_entropy_generator = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.ones_like(discriminator_fakelogits),
        logits = discriminator_fakelogits)
    )

gradient_penalty_weight = 10.0

discriminator_gradnorm = tf.sqrt(tf.math.reduce_sum(tf.square(tf.gradients(discriminator_gradlogits, discriminator_gradinput)[0]), axis=[1,2]))
gradient_penalty = gradient_penalty_weight * tf.reduce_mean((discriminator_gradnorm - 1)**2, axis=0)
summary_gradient_penatly = tf.summary.scalar("gradient_penalty", gradient_penalty / gradient_penalty_weight)

#dloss = cross_entropy_discriminator + gradient_penalty
dloss = - wasserstein_distance + gradient_penalty
summary_dloss = tf.summary.scalar("loss_discriminator", dloss)
dopt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1 = 0.5, beta2 = 0.9)
dtrainstep = dopt.minimize(dloss, var_list = discriminatorvariables)

#gloss = cross_entropy_generator
gloss = - tf.reduce_mean(discriminator_fakelogits)
summary_gloss = tf.summary.scalar("loss_generator", gloss)
gopt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1 = 0.5, beta2 = 0.9)
gtrainstep = gopt.minimize(gloss, var_list = generatorvariables)


# ========
# training
# ========
batchsize = 1024
discriminator_steps = 5
epochs = 15000


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    timestamp = str(datetime.now()).replace(":", "-").replace(" ","-")
    logdir = os.path.join("./tensorboard/", timestamp)
    os.mkdir(logdir)
    sumwriter = tf.summary.FileWriter(logdir, graph=sess.graph)

    globalsteps_d = 0
    globalsteps_g = 0

    starttime = timeit.default_timer()

    try:
        for r in range(epochs):
            print()
            log.info(f"starting epoch {r+1}/{epochs}")
    
            for x_batch, y_batch, n_batch, curstep, numsteps in yieldbatch(data[testlen:], label[testlen:], batchsize):
                if curstep % discriminator_steps != 0:
                    sessin = [
                            summary_dloss,
                            summary_wasserstein_distance,
                            summary_gradient_penatly,
                            dtrainstep,
                    ]

                    sessout = sess.run(
                            sessin,
                            feed_dict = {
                                xin: x_batch,
                                yin: y_batch,
                                nin: n_batch,
                            })

                    globalsteps_d += 1
                    sumwriter.add_summary(sessout[0], global_step = globalsteps_d)
                    sumwriter.add_summary(sessout[1], global_step = globalsteps_d)
                    sumwriter.add_summary(sessout[2], global_step = globalsteps_d)

                else:
                    sessin = [
                            summary_gloss,
                            gtrainstep,
                    ]

                    sessout = sess.run(
                            sessin,
                            feed_dict = {
                                yin: y_batch,
                                nin: n_batch,
                            })

                    globalsteps_g += 1
                    sumwriter.add_summary(sessout[0], global_step = globalsteps_g)
    
    except KeyboardInterrupt:
        pass

    runtime = timeit.default_timer() - starttime
    log.info(f"runtime: {runtime} seconds")

    for x_batch, y_batch, n_batch, curstep, numsteps in yieldbatch(data[0:testlen], label[0:testlen], testlen):
        log.info("generating test set")
        starttime = timeit.default_timer()
        xout = sess.run(generatoroutput,
            feed_dict = {
                yin: label[0:testlen],
                nin: n_batch,
        })
        np.save("gdata",  xout,             fix_imports=False)
        np.save("glabel", label[0:testlen], fix_imports=False)
        np.save("rdata",  data[0:testlen],  fix_imports=False)
        runtime = timeit.default_timer() - starttime
        log.info(f"runtime: {runtime} seconds")
        break

log.info("training done")

log.info("script done")

