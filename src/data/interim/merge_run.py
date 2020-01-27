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
    outpath = os.path.join(processedpath, run + "_merge")
    try:
        os.mkdir(outpath)
    except FileExistsError:
        if overwrite:
            pass
        else:
            log.error(f"processed run \"{run}\" does already exist")
            raise

    runpath = os.path.join(interimpath, run)
    if not os.path.isdir(runpath):
        msg = f"{runpath} does not exist"
        log.error(msg)
        raise IOError(msg)

    metadata_files = glob.glob(os.path.join(runpath, "metadata_*.json"))

    metadata = []
    for filepath in metadata_files:
        filename = os.path.split(filepath)[-1]
        with open(filepath, "r") as fp:
            metadata.append(json.load(fp))

    # make depth vectors
    depthfeatures = get_depth_features(metadata, expand_depth)

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

    # write data
    features = ["particle_distribution", "energy_deposit"]
    for feature in features:
        data = merge_data(metadata, feature, depthfeatures[feature]["len"], runpath)
        filepath = os.path.join(outpath, feature + ".npy")
        log.info("writing %s.npy to disk", feature)
        np.save(filepath, data, fix_imports=False)
        del(data)

    # write label
    label = merge_label(metadata, runpath)
    filepath = os.path.join(outpath, "label.npy")
    log.info("writing %s.npy to disk", "label")
    np.save(filepath, label, fix_imports=False)
    numdata = len(label)
    del(label)

    # write metadata
    meta = create_metadata(metadata, run, numdata, expand_depth)
    meta["particle_distribution"]["depth"] = depth_pd.tolist()
    meta["energy_deposit"]["depth"] = depth_ed.tolist()
    filepath = os.path.join(outpath, "metadata.json")
    log.info("writing metadata.json to disk")
    with open(filepath, "w") as fp:
        json.dump(meta, fp, indent=4)


def merge_data(metadata, feature, depthlen, runpath):
    log.info("merging numpy files for %s", feature)

    # get numdata
    numdata = 0
    for value in metadata:
        numdata += value["length"]

    numchannel = metadata[0][feature]["number_of_features"]

    data = np.full((numdata, depthlen, numchannel), np.nan, dtype=np.float)
    curindex = 0
    for value in metadata:
        jsonfile = value["json_file"]
        timestamp = jsonfile.split("_")[-1].split(".json")[0]
        filepath = os.path.join(runpath, feature + "_" + timestamp + ".npy")
        temp = np.load(filepath)
        curlen = np.min([depthlen, temp.shape[1]])
        data[curindex:curindex+len(temp),0:curlen,:] = temp[:,0:curlen,:]
        curindex += len(temp)

    return data



def merge_label(metadata, runpath):
    log.info("merging numpy files for label")

    # get numdata
    numdata = 0
    for value in metadata:
        numdata += value["length"]

    numchannel = metadata[0]["label"]["number_of_features"]

    data = np.full((numdata, numchannel), np.nan, dtype=np.float)
    curindex = 0
    for value in metadata:
        jsonfile = value["json_file"]
        timestamp = jsonfile.split("_")[-1].split(".json")[0]
        filepath = os.path.join(runpath, "label_" + timestamp + ".npy")
        temp = np.load(filepath)
        data[curindex:curindex+len(temp),:] = temp
        curindex += len(temp)

    return data



def create_metadata(metadata, run, numdata, expand_depth):
    log.info("creating metadata")

    jsonfiles = []
    expand_depths = []
    for value in metadata:
        jsonfiles.append(value["json_file"])
        expand_depths.append(value["expand_depth"])

    meta = {
            "json_files": list(jsonfiles),
            "run": run,
            "expand_depth": expand_depth,
            "expand_depth_list": list(expand_depths),
            "length": numdata,

            "label": dict(metadata[0]["label"]),
            "particle_distribution": dict(metadata[0]["particle_distribution"]),
            "energy_deposit": dict(metadata[0]["energy_deposit"]),
            }

    return meta



def get_depth_features(metadata, expand_depth):
    temp = np.zeros((2, 3, len(metadata)), dtype=np.int)
    for ii, value in enumerate(metadata):
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

    return depthfeatures

