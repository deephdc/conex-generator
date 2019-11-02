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
    data_pd = np.zeros(
            [
                numdata,
                depthfeatures["particle_distribution"]["len"],
                len(numpy_data_layout["particle_distribution"]),
            ])

    data_ed = np.zeros(
            [
                numdata,
                depthfeatures["energy_deposit"]["len"],
                len(numpy_data_layout["energy_deposit"]),
            ])

    # create label placeholder
    label = np.zeros(
            [
                numdata,
                len(numpy_label_layout),
            ])

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

        # write energy deposit
        curfeature = "energy_deposit"
        curdepthlen = np.min(
                [
                    len(value[curfeature]["depth"]),
                    depthfeatures[curfeature]["len"],
                ])
        for curkey, curindex in numpy_data_layout[curfeature].items():
            data_ed[ii, 0:curdepthlen, curindex] = value[curfeature][curkey][0:curdepthlen]

        # write labels
        for curkey, curindex in numpy_label_layout.items():
            if curkey == "particle":
                label[ii, curindex] = particle_label[value["particle"]]
                continue

            if curkey == "obslevel":
                try:
                    label[ii, curindex] = value[curkey]
                except:
                    pass
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
                "layout": numpy_data_layout["particle_distribution"],
                "number_of_features": data_pd.shape[2],
                },

            "energy_deposit": {
                "depth": depth_ed.tolist(),
                "data": data_ed,
                "layout": numpy_data_layout["energy_deposit"],
                "number_of_features": data_ed.shape[2],
                },
            }

