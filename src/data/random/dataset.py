import tensorflow as tf
import numpy as np

import src.utils
log = src.utils.getLogger(__name__)


def uniform_dataset(shape=(1,), minval=-1.0, maxval=1.0, dtype=np.float32,
                    buffsize=131072):
    """Return TensorFlow dataset with uniform distribution.

    Parameters
    ----------
    shape : tuple, optional
        Shape of a random sample. Defaults to (1,).
    minval : float, optional
        Lower bound of the distribution. Defaults to -1.0.
    maxval : float, optional
        Upper bound of the distribution. Defaults to +1.0.
    dtype : np.dtype, optional
        Data type which should be served. Defaults to np.float32.
    buffsize : int
        Size of the internal buffer which is sampled at once.
        Defaults to 131072.

    Returns
    -------
    ds : tf.data.Dataset
        TensorFlow dataset.
    """
    ds = tf.data.Dataset.from_generator(
            uniform_generator(dtype), output_types=dtype,
            output_shapes=(buffsize, *shape),
            args=[(buffsize, *shape), minval, maxval])
    ds = ds.unbatch().repeat()
    return ds

def uniform_generator_float32(shape, minval, maxval, runs=1024):
    """Return a float32 generator, which yields samples from a uniform distribution.

    Parameters
    ----------
    shape : tuple
        Shape of a random sample.
    minval : float
        Lower bound of the distribution.
    maxval : float
        Upper bound of the distribution.
    runs : int, optional
        Number of runs before the generator is exhausted. Defaults to 1024.

    Returns
    -------
    generator
    """
    for ii in range(runs):
        yield tf.random.uniform(shape=shape, minval=minval, maxval=maxval,
                                dtype=tf.float32)

def uniform_generator(dtype):
    """Select appropriate generator dtype.
    
    Raises exception if dtype is not supported.

    Paramters:
    ----------
    dtype : np.dtype
        Data type which should be served.

    Returns
    -------
    generator
    """
    if dtype == tf.float32:
        return uniform_generator_float32
    
    msg = "uniform random generator does not support this dtype"
    log.critical(msg)
    raise NotImplementedError(msg)


def normal_dataset(shape=(1,), mean=0.0, stddev=1.0, dtype=np.float32,
                   buffsize=131072):
    """Return TensorFlow dataset with normal distribution.

    Parameters
    ----------
    shape : tuple, optional
        Shape of a random sample. Defaults to (1,).
    mean : float, optional
        Mean of the distribution. Defaults to 0.0.
    stddev : float, optional
        Standard deviation of the distribution. Defaults to 1.0.
    dtype : np.dtype, optional
        Data type which should be served. Defaults to np.float32.
    buffsize : int
        Size of the internal buffer which is sampled at once.
        Defaults to 131072.

    Returns
    -------
    ds : tf.data.Dataset
        TensorFlow dataset.
    """
    ds = tf.data.Dataset.from_generator(
            normal_generator(dtype), output_types=dtype,
            output_shapes=(buffsize, *shape),
            args=[(buffsize, *shape), mean, stddev])
    ds = ds.unbatch().repeat()
    return ds

def normal_generator_float32(shape, mean, stddev, runs=1024):
    """Return a float32 generator, which yields samples from a normal distribution.

    Parameters
    ----------
    shape : tuple
        Shape of a random sample.
    mean : float
        Mean of the distribution.
    stddev : float
        Standard deviation of the distribution.
    runs : int, optional
        Number of runs before the generator is exhausted. Defaults to 1024.

    Returns
    -------
    generator
    """
    for ii in range(runs):
        yield tf.random.normal(shape=shape, mean=mean, stddev=stddev,
                               dtype=tf.float32)

def normal_generator(dtype):
    """Select appropriate generator dtype.
    
    Raises exception if dtype is not supported.

    Paramters:
    ----------
    dtype : np.dtype
        Data type which should be served.

    Returns
    -------
    generator
    """
    if dtype == tf.float32:
        return normal_generator_float32
    
    msg = "normal random generator does not support this dtype"
    log.critical(msg)
    raise NotImplementedError(msg)


def truncated_normal_dataset(shape=(1,), mean=0.0, stddev=1.0, dtype=np.float32,
                             buffsize=131072):
    """Return TensorFlow dataset with truncated normal distribution.

    Parameters
    ----------
    shape : tuple, optional
        Shape of a random sample. Defaults to (1,).
    mean : float, optional
        Mean of the distribution. Defaults to 0.0.
    stddev : float, optional
        Standard deviation of the distribution. Defaults to 1.0.
    dtype : np.dtype, optional
        Data type which should be served. Defaults to np.float32.
    buffsize : int
        Size of the internal buffer which is sampled at once.
        Defaults to 131072.

    Returns
    -------
    ds : tf.data.Dataset
        TensorFlow dataset.
    """
    ds = tf.data.Dataset.from_generator(
            truncated_normal_generator(dtype), output_types=dtype,
            output_shapes=(buffsize, *shape),
            args=[(buffsize, *shape), mean, stddev])
    ds = ds.unbatch().repeat()
    return ds

def truncated_normal_generator_float32(shape, mean, stddev, runs=1024):
    """Return a float32 generator, which yields samples from a truncated normal distribution.

    Parameters
    ----------
    shape : tuple
        Shape of a random sample.
    mean : float
        Mean of the distribution.
    stddev : float
        Standard deviation of the distribution.
    runs : int, optional
        Number of runs before the generator is exhausted. Defaults to 1024.

    Returns
    -------
    generator
    """
    for ii in range(runs):
        yield tf.random.truncated_normal(shape=shape, mean=mean, stddev=stddev,
                                         dtype=tf.float32)

def truncated_normal_generator(dtype):
    """Select appropriate generator dtype.
    
    Raises exception if dtype is not supported.

    Paramters:
    ----------
    dtype : np.dtype
        Data type which should be served.

    Returns
    -------
    generator
    """
    if dtype == tf.float32:
        return truncated_normal_generator_float32
    
    msg = "truncated normal random generator does not support this dtype"
    log.critical(msg)
    raise NotImplementedError(msg)

