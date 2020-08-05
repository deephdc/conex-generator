import numpy as np
import os
import glob

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
rawpath = get_path()

from . import json_to_numpy


def convert_run(run, expand_depth, overwrite=False):
    """Process raw data json files to numpy files and store metadata.

    This function searches for all .json files in the right timestamped format
    and process them to corresponding json files. Also metadata statistics
    are generated and stored alongside.

    For each conex_data_timestamp.json file the following files will be
    created:
        particle_distribution_timestamp.py
        energy_deposit_timestamp.py
        label_timestamp.py
        metadata_timestamp.py

    Parameters
    ----------
    run : str
        Name of the data/raw subfolder that should be processed.
    expand_depth : bool
        Flag to indicate if different size longitudinal profiles (due to
        different zeniths) should be expanded (True) to maximum size with
        additional nan values or if all profiles should be cut (False) to
        minimum size. This happens for each file separately.
    overwrite : bool, optional
        Flag to indicate if already processed files should be overwritten.
        Raises exception if files cannot be overwritten. Defaults to False.
    """
    runpath = os.path.join(rawpath, run)

    if not os.path.isdir(runpath):
        msg = f"{runpath} does not exist"
        log.error(msg)
        raise IOError(msg)

    json_files = glob.glob(os.path.join(runpath, "*.json"))

    for filepath in json_files:
        filename = os.path.split(filepath)[-1]
        dataobject = json_to_numpy.create_dataobject(filename, run, expand_depth)
        json_to_numpy.store_dataobject(dataobject, overwrite)

