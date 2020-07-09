import numpy as np
import glob
import os
import shutil
import json

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
interimpath = get_path()

from ..processed import get_path as get_processed_path
processedpath = get_processed_path()

from .merge_run import get_depth_features
from .merge_run import create_metadata


def align_run(run, expand_depth, overwrite=False):
    outpath = os.path.join(processedpath, run)
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
        log.info("writing %s*.npy files to disk", feature)
        for meta in metadata:
            data = expand_data(meta, feature, depthfeatures[feature]["len"], runpath)
            jsonfile = meta["json_file"]
            timestamp = jsonfile.split("_")[-1].split(".json")[0]
            filepath = os.path.join(outpath, feature + "_" + timestamp + ".npy")
            np.save(filepath, data, fix_imports=False)
            del(data)

    # write label
    log.info("copying %s.npy files", "label")
    for meta in metadata:
        copy_label(meta, runpath, outpath)

    # write cutbin
    log.info("copying %s.npy files", "cutbin")
    for meta in metadata:
        copy_cutbin(meta, runpath, outpath)

    # write metadata
    meta = create_metadata(metadata, run, expand_depth)
    meta["particle_distribution"]["depth"] = depth_pd.tolist()
    meta["energy_deposit"]["depth"] = depth_ed.tolist()
    filepath = os.path.join(outpath, "metadata.json")
    log.info("writing metadata.json to disk")
    with open(filepath, "w") as fp:
        json.dump(meta, fp, indent=4)


def expand_data(meta, feature, depthlen, runpath):
    numdata = meta["length"]
    numchannel = meta[feature]["number_of_features"]

    data = np.full((numdata, depthlen, numchannel), np.nan, dtype=np.float)

    jsonfile = meta["json_file"]
    timestamp = jsonfile.split("_")[-1].split(".json")[0]
    filepath = os.path.join(runpath, feature + "_" + timestamp + ".npy")

    temp = np.load(filepath)
    assert data.shape[0] == temp.shape[0]
    assert data.shape[2] == temp.shape[2]

    curlen = np.min([depthlen, temp.shape[1]])
    data[:,0:curlen,:] = temp[:,0:curlen,:]

    return data


def copy_label(meta, runpath, outpath):
    jsonfile = meta["json_file"]
    timestamp = jsonfile.split("_")[-1].split(".json")[0]
    filepath = os.path.join(runpath, "label" + "_" + timestamp + ".npy")
    copypath = os.path.join(outpath, "label" + "_" + timestamp + ".npy")
    shutil.copyfile(filepath, copypath)


def copy_cutbin(meta, runpath, outpath):
    jsonfile = meta["json_file"]
    timestamp = jsonfile.split("_")[-1].split(".json")[0]

    # copy pd cutbin
    filepath = os.path.join(runpath, "cutbin_particle_distribution" + "_" + timestamp + ".npy")
    copypath = os.path.join(outpath, "cutbin_particle_distribution" + "_" + timestamp + ".npy")
    shutil.copyfile(filepath, copypath)

    # copy ed cutbin
    filepath = os.path.join(runpath, "cutbin_energy_deposit" + "_" + timestamp + ".npy")
    copypath = os.path.join(outpath, "cutbin_energy_deposit" + "_" + timestamp + ".npy")
    shutil.copyfile(filepath, copypath)

