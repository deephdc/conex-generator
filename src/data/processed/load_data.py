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
    if batchread == 0:
        return line_generator(filepath)
    else:
        return batch_generator(filepath, batchread)

def line_generator(filepath):
    array = np.load(filepath, mmap_mode="c", fix_imports=False)
    return (data for data in array)

def batch_generator(filepath, batchread):
    array = np.load(filepath, mmap_mode="c", fix_imports=False)
    length = array.shape[0]
    if batchread < 0:
        batchread = int(np.ceil(-length/batchread))

    for ii in range(0, length, batchread):
        yield array[ii:ii+batchread]


def concatenate_datasets(datasets):
    ds = datasets[0]
    for ii in range(len(datasets)-1):
        ds = ds.concatenate(datasets[ii+1])
    return ds

