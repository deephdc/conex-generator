import tensorflow as tf
import numpy as np
import os
import json

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
curpath = get_path()


def load_data(run):
    check_files(run)
    runpath = os.path.join(curpath, run)

    particle_distribution = np.load(
            os.path.join(runpath, "particle_distribution.npy"))

    energy_deposit = np.load(
            os.path.join(runpath, "energy_deposit.npy"))

    label = np.load(
            os.path.join(runpath, "label.npy"))

    with open(os.path.join(runpath, "metadata.json"), "r") as fp:
        metadata = json.load(fp)

    particle_distribution = tf.data.Dataset.from_tensor_slices(particle_distribution)
    energy_deposit = tf.data.Dataset.from_tensor_slices(energy_deposit)
    label = tf.data.Dataset.from_tensor_slices(label)

    tempds = tf.data.Dataset.zip((particle_distribution, energy_deposit))
    dataset = tf.data.Dataset.zip((tempds, label))

    return (dataset, metadata)



def check_files(run):
    runpath = os.path.join(curpath, run)
    if not os.path.isdir(runpath):
        msg = f"run \"{run}\" does not exist"
        log.error(msg)
        raise IOError(msg)

    filepath_pd = os.path.join(runpath, "particle_distribution.npy")
    filepath_ed = os.path.join(runpath, "energy_deposit.npy")
    filepath_label = os.path.join(runpath, "label.npy")
    filepath_metadata = os.path.join(runpath, "metadata.json")

    filepaths = (filepath_pd, filepath_ed, filepath_label, filepath_metadata)
    for path in filepaths:
        if not os.path.isfile(path):
            filename = os.path.split(path)[-1]
            msg = f"file \"{filename}\" does not exist in run \"{run}\""
            log.error(msg)
            raise IOError(msg)

