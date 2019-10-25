import numpy as np
import os
import json

from ...interim import get_path
interimpath = get_path()


def store_dataobject(dataobject, overwrite=False):
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
        if os.path.isfile(curfilepath):
            if not overwrite:
                msg = f"{curfilepath} already exists - cannot overwrite"
                raise FileExistsError(msg)
            else:
                os.remove(curfilepath)
        np.save(curfilepath, dataobject[curfeature].pop("data"),
                fix_imports=False)

    # write metadata
    curfile = "metadata_" + timestamp + ".json"
    curfilepath = os.path.join(runpath, curfile)
    if os.path.isfile(curfilepath):
        if not overwrite:
            msg = f"{curfilepath} already exists - cannot overwrite"
            raise FileExistsError(msg)
        else:
            os.remove(curfilepath)
    with open(curfilepath, "w") as fd:
        json.dump(dataobject, fd, indent=4, sort_keys=False)

