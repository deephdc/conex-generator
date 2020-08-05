import numpy as np
import os
import json

import src.utils
log = src.utils.getLogger(__name__)

from ...interim import get_path
interimpath = get_path()


def store_dataobject(dataobject, overwrite=False):
    """Save dataobject to numpy files and metadata file.

    This functions intends to write the (converted) contents of a json file
    to their corresponding numpy and metadata file format in data/interim.
    The following files will be created:
        particle_distribution_timestamp.py
        energy_deposit_timestamp.py
        label_timestamp.py
        metadata_timestamp.json

    Parameters
    ----------
    dataobject : dict
        Dictionary that contains all json file information. (See also
        src.data.raw.json_to_numpy.create_dataobject)
    overwrite : bool, optional
        Flag to indicate if files in data/interim should be overwritten.
        Raises exception if overwrite fails.
    """
    json_file : str = dataobject["json_file"]
    run : str = dataobject["run"]
    timestamp = json_file.split("conex_data_")[-1].split(".json")[0]

    runpath = os.path.join(interimpath, run)
    try:
        os.mkdir(runpath)
    except FileExistsError:
        pass

    # write npy files
    features = ["particle_distribution", "energy_deposit", "label"]
    for curfeature in features:
        curfile = curfeature + "_" + timestamp + ".npy"
        curfilepath = os.path.join(runpath, curfile)
        log.info("writing %s to disk", curfile)
        if os.path.isfile(curfilepath):
            if not overwrite:
                msg = f"{curfilepath} already exists - cannot overwrite"
                log.error(msg)
                raise FileExistsError(msg)
            else:
                msg = f"overwrite of {curfilepath}"
                log.warning(msg)
                os.remove(curfilepath)
        np.save(curfilepath, dataobject[curfeature].pop("data"),
                fix_imports=False)

    # write cutbin files
    features = ["particle_distribution", "energy_deposit"]
    for curfeature in features:
        curfile = "cutbin_" + curfeature + "_" + timestamp + ".npy"
        curfilepath = os.path.join(runpath, curfile)
        log.info("writing %s to disk", curfile)
        if os.path.isfile(curfilepath):
            if not overwrite:
                msg = f"{curfilepath} already exists - cannot overwrite"
                log.error(msg)
                raise FileExistsError(msg)
            else:
                msg = f"overwrite of {curfilepath}"
                log.warning(msg)
                os.remove(curfilepath)
        np.save(curfilepath, dataobject[curfeature].pop("cutbin"),
                fix_imports=False)

    # write metadata
    curfile = "metadata_" + timestamp + ".json"
    curfilepath = os.path.join(runpath, curfile)
    log.info("writing %s to disk", curfile)
    if os.path.isfile(curfilepath):
        if not overwrite:
            msg = f"{curfilepath} already exists - cannot overwrite"
            log.error(msg)
            raise FileExistsError(msg)
        else:
            msg = f"overwrite of {curfilepath}"
            log.warning(msg)
            os.remove(curfilepath)
    with open(curfilepath, "w") as fd:
        json.dump(dataobject, fd, indent=4, sort_keys=False)

