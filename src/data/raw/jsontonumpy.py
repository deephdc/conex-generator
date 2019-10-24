import os
import json
import numpy as np
import sys

import src.utils
_log = src.utils.getLogger(__name__)

_particle_label = {
        "gamma":    0,
        "electron": 1,
        "proton":   2,
        "helium":   3,
        "oxygen":   4,
        "iron":     5,
        }

_numpy_data_layout = {
        "particle_distribution": {
            "gamma":    0,
            "positron": 1,
            "electron": 2,
            "mup":      3,
            "mum":      4,
            "hadron":   5,
            "charged":  6,
            "nuclei":   7,
            },

        "energy_deposit": {
            "gamma":    0,
            "em_ioniz": 1,
            "em_cut":   2,
            "mu_ioniz": 3,
            "mu_cut":   4,
            "ha_ioniz": 5,
            "ha_cut":   6,
            "neutrino": 7,
            "sum":      8,
            }
        }

_numpy_label_layout = {
        "particle": 0,
        "energy":   1,
        "theta":    2,
        "phi":      3,
        }



def create_dataobject(
        filename,
        expand_depth = True
        datapath = "~/data/network/conex-generator/data",
        run = "run02",
        ):

    # read json data
    filepath = os.path.join(datapath, "raw", run, filename)
    with open(filepath, "r") as fd:
        _log.info("reading json data: %s", filepath)
        jsondata = json.load(fd)

    _log.info("getting depth features")
    numdata = len(jsondata)
    temp = np.zeros((2, 3, numdata))
    for ii, values in enumerate(jsondata.values()):
        temp[0,0,ii] = len(value["particle_distribution"]["depth"])
        temp[0,1,ii] = np.min(value["particle_distribution"]["depth"])
        temp[0,2,ii] = np.max(value["particle_distribution"]["depth"])

        temp[1,0,ii] = len(value["energy_deposit"]["depth"])
        temp[1,1,ii] = np.min(value["energy_deposit"]["depth"])
        temp[1,2,ii] = np.max(value["energy_deposit"]["depth"])

    if expand_depth:
        _log.info("expanding to maximum depth")
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
        _log.info("shrinking to minimum depth")
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
                len(_numpy_data_layout["particle_distribution"]),
            ])

    data_ed = np.zeros(
            [
                numdata,
                depthfeatures["energy_deposit"]["len"],
                len(_numpy_data_layout["energy_deposit"]),
            ])

    # create label placeholder
    label = np.zeros(
            [
                numdata,
                len(_numpy_label_layout),
            ])

    for ii, value in enumerate(jsondata.values()):
        # write particle distribution
        curfeature = "particle_distribution"
        curdepthlen = np.min(
                [
                    len(value[curfeature]["depth"]),
                    depthfeatures[curfeature]["len"],
                ])
        for curkey, curindex in _numpy_data_layout[curfeature].items():
            data_pd[ii, 0:curdepthlen, curindex] = value[curfeature][curkey][0:curdepthlen]

        # write energy deposit
        curfeature = "energy_deposit"
        curdepthlen = np.min(
                [
                    len(value[curfeature]["depth"]),
                    depthfeatures[curfeature]["len"],
                ])
        for curkey, curindex in _numpy_data_layout[curfeature].items():
            data_pd[ii, 0:curdepthlen, curindex] = value[curfeature][curkey][0:curdepthlen]

        # write labels
        for curkey, curindex in _numpy_label_layout.items():
            if curkey == "particle":
                label[ii, curindex] = _particle_label[value["particle"]]
            else:
                label[ii, curindex] = value[curkey]

        return {
                "label": {
                    "layout": _numpy_label_layout,
                    "particle": _particle_label,
                    },

                "particle_distribution": {
                    "depth": depth_pd,
                    "data": data_pd,
                    "layout": _numpy_data_layout["particle_distribution"],
                    },

                "energy_deposit": {
                    "depth": depth_ed,
                    "data": data_pd,
                    "layout": _numpy_data_layout["energy_deposit"],
                    },
                }

# write numpy data
"""
print("writing numpy data to disk")
filename = os.path.join(data_path, "data.npy")
np.save(filename, data, fix_imports=False)
filename = os.path.join(data_path, "label.npy")
np.save(filename, label, fix_imports=False)
"""
