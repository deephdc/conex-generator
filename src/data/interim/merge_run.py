import numpy as np
import glob
import os
import json

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
interimpath = get_path()

from ..processed import get_path as get_processed_path
processedpath = get_processed_path()


def merge_run(run, expand_depth, overwrite=False):
    runpath = os.path.join(interimpath, run)

    if not os.path.isdir(runpath):
        msg = f"{runpath} does not exist"
        log.error(msg)
        raise IOError(msg)

    metadata_files = glob.glob(os.path.join(runpath, "metadata_*.json"))

    metadata = []
    timestamp = []
    for filepath in metadata_files:
        filename = os.path.split(filepath)[-1]
        timestamp.append(filename.split("metadata_")[-1].split(".json")[0])
        with open(filepath, "r") as fp:
            metadata.append(json.load(fp))


