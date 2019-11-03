import json
import os
import src.utils

from . import get_path

log = src.utils.getLogger(__name__)


def store_data(dataobject, run):
    raw_path = get_path()
    runpath = os.path.join(raw_path, run)
    if not os.path.isdir(runpath):
        os.mkdir(runpath)

    timestamp = src.utils.timestamp()
    filepath = os.path.join(runpath, "conex_data_" + timestamp + ".json")
    filename = os.path.split(filepath)[-1]

    log.info(f"writing raw file {filename} to run {run}")
    with open(filepath, "w") as fp:
        json.dump(dataobject, fp)

