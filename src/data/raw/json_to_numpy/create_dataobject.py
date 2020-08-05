import os
import json
import numpy as np
import sys

import src.utils
log = src.utils.getLogger(__name__)


# get raw data path
from .. import get_path
rawpath = get_path()

# load config
script_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_path, "config.json"), "r") as fd:
    config = json.load(fd)

particle_label = config["particle_label"]
numpy_data_layout = config["numpy_data_layout"]
numpy_label_layout = config["numpy_label_layout"]


def create_dataobject(filename, run, expand_depth):
    """Create a dictionary from a json file and convert data to numpy.
    
    Parameters
    ----------
    filename : str
        Filename (without path) of the json file that should be converted.
    run : str
        Subfolder of data/raw frow which json files should be read.
    expand_depth : bool
        Flag to indicate if different size longitudinal profiles (due to
        different zeniths) should be expanded (True) to maximum size with
        additional nan values or if all profiles should be cut (False) to
        minimum size.

    Returns
    -------
    dataobject : dict
        Dictionary with metadata and converted data in numpy format.
    """
    # read json data
    filepath = os.path.join(rawpath, run, filename)
    with open(filepath, "r") as fd:
        log.info("reading json data: %s", filepath)
        jsondata = json.load(fd)

    # extract data features
    log.info("getting depth features")
    numdata = len(jsondata)
    temp = np.zeros((2, 3, numdata), dtype=np.int)
    for ii, value in enumerate(jsondata.values()):
        temp[0,0,ii] = len(value["particle_distribution"]["depth"])
        temp[0,1,ii] = np.min(value["particle_distribution"]["depth"])
        temp[0,2,ii] = np.max(value["particle_distribution"]["depth"])

        temp[1,0,ii] = len(value["energy_deposit"]["depth"])
        temp[1,1,ii] = np.min(value["energy_deposit"]["depth"])
        temp[1,2,ii] = np.max(value["energy_deposit"]["depth"])

    # shrink or expand to same depth length
    if expand_depth:
        log.info("expanding to maximum depth")
        depthfeatures = {
                "particle_distribution": {
                    "len": np.max(temp[0,0,:]),
                    "min": np.min(temp[0,1,:]),
                    "max": np.max(temp[0,2,:]),
                    },
                "energy_deposit": {
                    "len": np.max(temp[1,0,:]),
                    "min": np.min(temp[1,1,:]),
                    "max": np.max(temp[1,2,:]),
                    },
                }

    else:
        log.info("shrinking to minimum depth")
        depthfeatures = {
                "particle_distribution": {
                    "len": np.min(temp[0,0,:]),
                    "min": np.min(temp[0,1,:]),
                    "max": np.min(temp[0,2,:]),
                    },
                "energy_deposit": {
                    "len": np.min(temp[1,0,:]),
                    "min": np.min(temp[1,1,:]),
                    "max": np.min(temp[1,2,:]),
                    },
                }

    # create general depth vector
    depth_pd = np.linspace(
            depthfeatures["particle_distribution"]["min"],
            depthfeatures["particle_distribution"]["max"],
            depthfeatures["particle_distribution"]["len"]
            )

    depth_ed = np.linspace(
            depthfeatures["energy_deposit"]["min"],
            depthfeatures["energy_deposit"]["max"],
            depthfeatures["energy_deposit"]["len"]
            )

    # create data placeholders
    data_pd = np.full(
            [
                numdata,
                depthfeatures["particle_distribution"]["len"],
                len(numpy_data_layout["particle_distribution"]),
            ],
            np.nan,
            dtype=np.float)

    data_ed = np.full(
            [
                numdata,
                depthfeatures["energy_deposit"]["len"],
                len(numpy_data_layout["energy_deposit"]),
            ],
            np.nan,
            dtype=np.float)

    # create cutbin placeholders
    data_pd_cut = np.full(
            [
                numdata,
                2,
                len(numpy_data_layout["particle_distribution"]),
            ],
            np.nan,
            dtype=np.float)

    data_ed_cut = np.full(
            [
                numdata,
                2,
                len(numpy_data_layout["energy_deposit"]),
            ],
            np.nan,
            dtype=np.float)

    # create label placeholder
    label = np.full(
            [
                numdata,
                len(numpy_label_layout),
            ],
            np.nan,
            dtype=np.float)

    # data processing loop
    log.info("writing json values to numpy")
    for ii, value in enumerate(jsondata.values()):
        # write particle distribution
        curfeature = "particle_distribution"
        curdepthlen = np.min(
                [
                    len(value[curfeature]["depth"]),
                    depthfeatures[curfeature]["len"],
                ])
        for curkey, curindex in numpy_data_layout[curfeature].items():
            data_pd[ii, 0:curdepthlen, curindex] = value[curfeature][curkey][0:curdepthlen]
            if "cutbin" in value:
                data_pd_cut[ii, :, curindex] = value["cutbin"][curfeature][curkey][:]


        # write energy deposit
        curfeature = "energy_deposit"
        curdepthlen = np.min(
                [
                    len(value[curfeature]["depth"]),
                    depthfeatures[curfeature]["len"],
                ])
        for curkey, curindex in numpy_data_layout[curfeature].items():
            data_ed[ii, 0:curdepthlen, curindex] = value[curfeature][curkey][0:curdepthlen]
            if "cutbin" in value:
                data_ed_cut[ii, :, curindex] = value["cutbin"][curfeature][curkey][:]

        # write labels
        for curkey, curindex in numpy_label_layout.items():
            if curkey == "particle":
                label[ii, curindex] = particle_label[value["particle"]]
                continue

            if curkey == "obslevel":
                try:
                    label[ii, curindex] = value[curkey]
                except:
                    label[ii, curindex] = np.nan
                continue

            label[ii, curindex] = value[curkey]

    return {
            "json_file": filename,
            "run": run,
            "expand_depth": expand_depth,
            "length": numdata,

            "label": {
                "data": label,
                "layout": numpy_label_layout,
                "particle": particle_label,
                "number_of_features": label.shape[1],
                },

            "particle_distribution": {
                "depth": depth_pd.tolist(),
                "data": data_pd,
                "cutbin": data_pd_cut,
                "layout": numpy_data_layout["particle_distribution"],
                "number_of_features": data_pd.shape[2],
                "max_data": np.nanmax(data_pd, axis=(0,1)).tolist(),
                "min_depthlen": int(np.min(temp[0,0,:])),
                },

            "energy_deposit": {
                "depth": depth_ed.tolist(),
                "data": data_ed,
                "cutbin": data_ed_cut,
                "layout": numpy_data_layout["energy_deposit"],
                "number_of_features": data_ed.shape[2],
                "max_data": np.nanmax(data_ed, axis=(0,1)).tolist(),
                "min_depthlen": int(np.min(temp[1,0,:])),
                },
            }

