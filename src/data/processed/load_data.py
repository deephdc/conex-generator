import tensorflow as tf
import numpy as np
import os
import glob
import json

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
curpath = get_path()


def load_data(run, memorymap=True, batchread=-5):
    """Load a given subfolder in data/processed as tf.data.Dataset.

    Parameters
    ----------
    run : str
        Subfolder name in data/processed from which the dataset should be
        generated.
    memorymap : bool
        Flag to indicate if the data should be memory mapped from
        filesystem (True) or not (False). Defaults to True.
    batchread : int
        Integer to define the size and mode of reading batches. This is only
        for internal speed considerations. The returned dataset will always
        be in unbatched form.
            = 0 --> don't read batches and return single data lines
            < 0 --> batchsize = np.ceil(length / -batchread)
            > 0 --> batchsize = batchread
        Defaults to -5 (1 file = 5 batches).

    Returns
    -------
    datasets : tuple
        Tuple of datasets in this order:
            particle distribution -> tf.data.Dataset
            energy deposit -> tf.data.Dataset
            label -> tf.data.Dataset
            metadata -> dict
        All tf.data.Dataset instances are unbatched and properly aligned.
        The data type depends on the stored numpy files (which is np.float
        by default, i.e. 64 bit floating point on current cpus).
    """
    runpath = os.path.join(curpath, run)
    check_files(runpath)

    filepath_pd = os.path.join(runpath, "particle_distribution*.npy")
    filepath_ed = os.path.join(runpath, "energy_deposit*.npy")
    filepath_label = os.path.join(runpath, "label*.npy")

    files_pd = glob.glob(filepath_pd)
    files_ed = glob.glob(filepath_ed)
    files_label = glob.glob(filepath_label)

    particle_distribution = []
    for f in files_pd:
        ds = load_dataset(f, memorymap, batchread)
        particle_distribution.append(ds)
    particle_distribution = concatenate_datasets(particle_distribution)

    energy_deposit = []
    for f in files_ed:
        ds = load_dataset(f, memorymap, batchread)
        energy_deposit.append(ds)
    energy_deposit = concatenate_datasets(energy_deposit)

    label = []
    for f in files_label:
        ds = load_dataset(f, memorymap, batchread)
        label.append(ds)
    label = concatenate_datasets(label)

    with open(os.path.join(runpath, "metadata.json"), "r") as fp:
        metadata = json.load(fp)

    return (particle_distribution, energy_deposit, label, metadata)


def check_files(runpath):
    """Execute checks on the files in data/processed.

    This function will first check for the existance of runpath and then check
    if there are the same numer of particle distribution, energy deposit and
    label files. It also checks if metadata.json is present. If one of the
    checks fails an exception is raised.
    
    Parameters
    ----------
    runpath : str
        Full path to the subfolder in data/processed.

    """
    rundir = os.path.split(runpath)[-1]
    if not os.path.isdir(runpath):
        msg = f"rundir \"{rundir}\" does not exist"
        log.error(msg)
        raise IOError(msg)

    filepath_pd = os.path.join(runpath, "particle_distribution*.npy")
    filepath_ed = os.path.join(runpath, "energy_deposit*.npy")
    filepath_label = os.path.join(runpath, "label*.npy")
    filepaths = (filepath_pd, filepath_ed, filepath_label)

    files_pd = glob.glob(filepath_pd)
    files_ed = glob.glob(filepath_ed)
    files_label = glob.glob(filepath_label)
    files = (files_pd, files_ed, files_label)

    assert len(files_pd) == len(files_ed)
    assert len(files_pd) == len(files_label)

    for f,fp in zip(files, filepaths):
        if len(f) == 0:
            filename = os.path.split(fp)[-1]
            msg = f"file \"{filename}\" does not exist in rundir \"{rundir}\""
            log.error(msg)
            raise IOError(msg)

    filepath_metadata = os.path.join(runpath, "metadata.json")
    if not os.path.isfile(filepath_metadata):
        filename = os.path.split(filepath_metadata)[-1]
        msg = f"file \"{filename}\" does not exist in rundir \"{rundir}\""
        log.error(msg)
        raise IOError(msg)


def load_dataset(filepath, memorymap, batchread):
    """Load a single file in data/processed as tf.data.Dataset.

    Parameters
    ----------
    filepath : str
        Full path to the file which should be loaded.
    memorymap : bool
        Flag to indicate if the numpy file should be memory mapped from the
        filesystem.
    batchread : int
        Integer to define the size and mode of reading batches. This is only
        for internal speed considerations. The returned dataset will always
        be in unbatched form.
            = 0 --> don't read batches and return single data lines
            < 0 --> batchsize = np.ceil(length / -batchread)
            > 0 --> batchsize = batchread
        Defaults to -5 (1 file = 5 batches).

    Returns
    -------
    ds : tf.data.Dataset
        TensorFlow dataset.
    """
    if memorymap:
        array = np.load(filepath, mmap_mode="c", fix_imports=False)
        if batchread == 0:
            ds = tf.data.Dataset.from_generator(
                    make_data_generator,
                    args=[filepath, batchread],
                    output_types=array.dtype,
                    output_shapes=array.shape[1:])
        else:
            ds = tf.data.Dataset.from_generator(
                    make_data_generator,
                    args=[filepath, batchread],
                    output_types=array.dtype,
                    output_shapes=(None, *array.shape[1:]))
            ds = ds.unbatch()
    else:
        array = np.load(filepath, fix_imports=False)
        ds = tf.data.Dataset.from_tensor_slices(array)

    return ds


def make_data_generator(filepath, batchread):
    """Return generator that serves data batches.
    
    Parameters
    ----------
    filepath : str
        Full path to the file which should be loaded.
    batchread : int
        Integer to define the size and mode of reading batches. This is only
        for internal speed considerations.
            = 0 --> don't read batches and return single data lines
            < 0 --> batchsize = np.ceil(length / -batchread)
            > 0 --> batchsize = batchread
        Defaults to -5 (1 file = 5 batches).

    Returns
    -------
    generator
        Either a single line or batch generator depending on the value of
        batchread.
    """
    if batchread == 0:
        return line_generator(filepath)
    else:
        return batch_generator(filepath, batchread)

def line_generator(filepath):
    """Return single line generator.

    Parameters
    ----------
    filepath : str
        Full path to the file which should be loaded.

    Returns
    -------
    generator
        Single line generator that serves slices of a numpy file.
    """
    array = np.load(filepath, mmap_mode="c", fix_imports=False)
    return (data for data in array)

def batch_generator(filepath, batchread):
    """Return batch generator.

    Parameters
    ----------
    filepath : str
        Full path to the file which should be loaded.
    batchread : int
        Integer to define the size and mode of reading batches.
            < 0 --> batchsize = np.ceil(length / -batchread)
            > 0 --> batchsize = batchread
        Defaults to -5 (1 file = 5 batches). Should NOT be 0!

    Returns
    -------
    generator
        Batch generator that serves slices of a numpy file.
    """
    array = np.load(filepath, mmap_mode="c", fix_imports=False)
    length = array.shape[0]
    if batchread < 0:
        batchread = int(np.ceil(-length/batchread))

    for ii in range(0, length, batchread):
        yield array[ii:ii+batchread]


def concatenate_datasets(datasets):
    """Concatenate a list of tf.data.Dataset.

    Parameters
    ----------
    datasets : list
        List of tf.data.Dataset

    Returns
    -------
    ds : tf.data.Dataset
        Concatenated TensorFlow dataset.
    """
    ds = datasets[0]
    for ii in range(len(datasets)-1):
        ds = ds.concatenate(datasets[ii+1])
    return ds

