import numpy as np
import os
import glob

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
rawpath = get_path()

from . import json_to_numpy


def convert_run(run, expand_depth, overwrite=False):
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

