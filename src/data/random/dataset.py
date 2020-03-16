import lazy_import

tf = lazy_import.lazy_module("tensorflow")
#import tensorflow as tf


def uniform_dataset(shape=(1,), minval=-1.0, maxval=1.0, dtype=tf.float32,
                    buffsize=131072):
    ds = tf.data.Dataset.from_generator(
            uniform_generator[dtype], output_types=dtype,
            output_shapes=(buffsize, *shape),
            args=[(buffsize, *shape), minval, maxval])
    ds = ds.unbatch().repeat()
    return ds

def uniform_generator_float32(shape, minval, maxval, runs=1024):
    for ii in range(runs):
        yield tf.random.uniform(shape=shape, minval=minval, maxval=maxval,
                                dtype=tf.float32)

uniform_generator = {
        tf.float32: uniform_generator_float32,
}


def normal_dataset(shape=(1,), mean=0.0, stddev=1.0, dtype=tf.float32,
                   buffsize=131072):
    ds = tf.data.Dataset.from_generator(
            normal_generator[dtype], output_types=dtype,
            output_shapes=(buffsize, *shape),
            args=[(buffsize, *shape), mean, stddev])
    ds = ds.unbatch().repeat()
    return ds

def normal_generator_float32(shape, mean, stddev, runs=1024):
    for ii in range(runs):
        yield tf.random.normal(shape=shape, mean=mean, stddev=stddev,
                               dtype=tf.float32)

normal_generator = {
        tf.float32: normal_generator_float32,
}


def truncated_normal_dataset(shape=(1,), mean=0.0, stddev=1.0, dtype=tf.float32,
                             buffsize=131072):
    ds = tf.data.Dataset.from_generator(
            truncated_normal_generator[dtype], output_types=dtype,
            output_shapes=(buffsize, *shape),
            args=[(buffsize, *shape), mean, stddev])
    ds = ds.unbatch().repeat()
    return ds

def truncated_normal_generator_float32(shape, mean, stddev, runs=1024):
    for ii in range(runs):
        yield tf.random.truncated_normal(shape=shape, mean=mean, stddev=stddev,
                                         dtype=tf.float32)
truncated_normal_generator = {
        tf.float32: truncated_normal_generator_float32,
}

