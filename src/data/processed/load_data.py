import tensorflow as tf
import numpy as np
import os
import glob
import json

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
curpath = get_path()


def load_data(run, memorymap=False):
    if memorymap:
        runpath = os.path.join(curpath, run + "_memorymap")
        mmap_mode = "r"
    else:
        runpath = os.path.join(curpath, run + "_memorymap")
        mmap_mode = None

    check_files(runpath)

    filepath_pd = os.path.join(runpath, "particle_distribution*.npy")
    filepath_ed = os.path.join(runpath, "energy_deposit*.npy")
    filepath_label = os.path.join(runpath, "label*.npy")

    files_pd = glob.glob(filepath_pd)
    files_ed = glob.glob(filepath_ed)
    files_label = glob.glob(filepath_label)

    particle_distribution = []
    for f in files_pd:
        curnumpy = np.load(f, mmap_mode=mmap_mode, fix_imports=False)
        ds = make_dataset(curnumpy, memorymap)
        particle_distribution.append(ds)
    particle_distribution = concatenate_datasets(particle_distribution)

    energy_deposit = []
    for f in files_ed:
        curnumpy = np.load(f, mmap_mode=mmap_mode, fix_imports=False)
        ds = make_dataset(curnumpy, memorymap)
        energy_deposit.append(ds)
    energy_deposit = concatenate_datasets(energy_deposit)

    label = []
    for f in files_label:
        curnumpy = np.load(f, mmap_mode=mmap_mode, fix_imports=False)
        ds = make_dataset(curnumpy, memorymap)
        label.append(ds)
    label = concatenate_datasets(label)


    tempds = tf.data.Dataset.zip((particle_distribution, energy_deposit))
    dataset = tf.data.Dataset.zip((tempds, label))

    with open(os.path.join(runpath, "metadata.json"), "r") as fp:
        metadata = json.load(fp)

    return (dataset, metadata)


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


def make_dataset(array, memorymap):
    if memorymap:
        ds = tf.data.Dataset.from_generator(
                data_generator,
                args=[array],
                output_types=array.dtype,
                output_shapes=array.shape[1:])
    else:
        ds = tf.data.Dataset.from_tensor_slices(array)

    return ds


def data_generator(array):
    return (data for data in array)


def concatenate_datasets(datasets):
    ds = datasets[0]
    for ii in range(len(datasets)-1):
        ds = ds.concatenate(datasets[ii+1])
    return ds

