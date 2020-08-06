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
    """Merge the contents of data/interim/run into a single dataset.

    Takes the seperately converted numpy and metadata files and merges them
    into a single file in data/processed, taking care of longitudinal profile
    size (expand_depth), i.e. all particle_distribution_timestamp.npy files
    will be merged into particle_distribution.npy. The subfolder path in
    data/processed will be appended with "_merge". src.data.clear_run will
    not take care of merge subfolders. They have to be deleted manually if
    necessary. The current data pipeline operates better with align_run than
    with merge_run.

    Parameters
    ----------
    run : str
        Subfolder of data/interim for which the merge should be done.
    expand_depth : bool
        Flag to indicate if different size longitudinal profiles (due to
        different zeniths) should be expanded (True) to maximum size with
        additional nan values or if all profiles should be cut (False) to
        minimum size. This happens for the whole dataset.
    overwrite : bool, optional
        Flag to indicate if already processed files should be overwritten.
        Raises exception if files cannot be overwritten. Defaults to False.
    """
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
    del(label)

    # write cutbin
    cut_pd, cut_ed = merge_cutbin(metadata, runpath)
    filepath_pd = os.path.join(outpath, "cutbin_particle_distribution.npy")
    filepath_ed = os.path.join(outpath, "cutbin_energy_deposit.npy")
    log.info("writing %s.npy to disk", "cutbin")
    np.save(filepath_pd, cut_pd, fix_imports=False)
    np.save(filepath_ed, cut_ed, fix_imports=False)
    del(cut_pd)
    del(cut_ed)

    # write metadata
    meta = create_metadata(metadata, run, expand_depth)
    meta["particle_distribution"]["depth"] = depth_pd.tolist()
    meta["energy_deposit"]["depth"] = depth_ed.tolist()
    filepath = os.path.join(outpath, "metadata.json")
    log.info("writing metadata.json to disk")
    with open(filepath, "w") as fp:
        json.dump(meta, fp, indent=4)


def merge_data(metadata, feature, depthlen, runpath):
    """Merge the different numpy files depending on the indicated feature.

    This function reads the numpy files in data/interim and merges them
    accordingly.

    Parameters
    ----------
    metadata : list
        List of dictionaries which contain the contents of
        metadata_timestamp.json.
    feature : str
        Data feature that should be merged. Can be "particle_distribution" or
        "energy_deposit".
    depthlen : int
        Longitudinal profile size in number of depth bins that should be used
        for the merged files.
    runpath : str
        Full subfolder path in data/interim on which merging is operated.

    Returns
    -------
    data : np.ndarray
        Numpy array that contains all data of the given feature type. Data
        might be cut down or padded with nan values depending on depthlen.
    """
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
    """Merge the different label numpy files.

    This function reads the label numpy files in data/interim and merges them
    accordingly.

    Parameters
    ----------
    metadata : list
        List of dictionaries which contain the contents of
        metadata_timestamp.json.
    runpath : str
        Full subfolder path in data/interim on which merging is operated.

    Returns
    -------
    data : np.ndarray
        Numpy array that contains all labels.
    """
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


def merge_cutbin(metadata, runpath):
    """Merge the different cutbin numpy files.

    This function reads the cutbin numpy files in data/interim and merges them
    accordingly.

    Parameters
    ----------
    metadata : list
        List of dictionaries which contain the contents of
        metadata_timestamp.json.
    runpath : str
        Full subfolder path in data/interim on which merging is operated.

    Returns
    -------
    data : (np.ndarray, np.ndarray)
        Tuple of numpy arrays that contains particle distribution and energy
        deposit cutbin data (in that order).
    """
    log.info("merging numpy files for cutbin")

    # get numdata
    numdata = 0
    for value in metadata:
        numdata += value["length"]

    # merge pd cutbin
    numchannel = metadata[0]["particle_distribution"]["number_of_features"]

    data_pd = np.full((numdata, 2, numchannel), np.nan, dtype=np.float)
    curindex = 0
    for value in metadata:
        jsonfile = value["json_file"]
        timestamp = jsonfile.split("_")[-1].split(".json")[0]
        filepath = os.path.join(runpath, "cutbin_particle_distribution_" + timestamp + ".npy")
        temp = np.load(filepath)
        data_pd[curindex:curindex+len(temp),:,:] = temp
        curindex += len(temp)

    # merge ed cutbin
    numchannel = metadata[0]["energy_deposit"]["number_of_features"]

    data_ed = np.full((numdata, 2, numchannel), np.nan, dtype=np.float)
    curindex = 0
    for value in metadata:
        jsonfile = value["json_file"]
        timestamp = jsonfile.split("_")[-1].split(".json")[0]
        filepath = os.path.join(runpath, "cutbin_energy_deposit_" + timestamp + ".npy")
        temp = np.load(filepath)
        data_ed[curindex:curindex+len(temp),:,:] = temp
        curindex += len(temp)

    return (data_pd, data_ed)


def create_metadata(metadata, run, expand_depth):
    """Merge the different metadata json files.

    This function reads the metadata json files in data/interim and merges them
    accordingly.

    Parameters
    ----------
    metadata : list
        List of dictionaries which contain the contents of
        metadata_timestamp.json.
    runpath : str
        Full subfolder path in data/interim on which merging is operated.

    Returns
    -------
    meta : dict
        New (merged) metadata dictionary.
    """
    log.info("creating metadata")

    jsonfiles = []
    expand_depths = []

    pd_max_data = []
    ed_max_data = []

    pd_min_depthlen = []
    ed_min_depthlen = []

    numdata = 0
    for value in metadata:
        jsonfiles.append(value["json_file"])
        expand_depths.append(value["expand_depth"])

        pd_max_data.append(value["particle_distribution"]["max_data"])
        ed_max_data.append(value["energy_deposit"]["max_data"])

        pd_min_depthlen.append(value["particle_distribution"]["min_depthlen"])
        ed_min_depthlen.append(value["energy_deposit"]["min_depthlen"])

        numdata += value["length"]

    pd_max_data = np.max(pd_max_data, axis=0).tolist()
    ed_max_data = np.max(ed_max_data, axis=0).tolist()

    pd_min_depthlen = int(np.min(pd_min_depthlen))
    ed_min_depthlen = int(np.min(ed_min_depthlen))

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

    meta["particle_distribution"]["max_data"] = pd_max_data
    meta["energy_deposit"]["max_data"] = ed_max_data

    meta["particle_distribution"]["min_depthlen"] = pd_min_depthlen
    meta["energy_deposit"]["min_depthlen"] = ed_min_depthlen

    return meta


def get_depth_features(metadata, expand_depth):
    """Handle expand_depth length calculation.

    This functions creates information (min, max) on the longitudinal profile
    lengths for handling the expand_depth flag (see also merge_run).

    Parameters
    ----------
    metadata : list
        List of dictionaries which contain the contents of
        metadata_timestamp.json.
    expand_depth : bool
        Flag to indicate if different size longitudinal profiles (due to
        different zeniths) should be expanded (True) to maximum size with
        additional nan values or if all profiles should be cut (False) to
        minimum size.

    Returns
    -------
    depthfeatures : dict
        Dictionary containing information about longitudinal profile size
        of all files in the dataset.
        Layout:
        {
            "particle_distribution": {
                "len": length in depthbins of the final data product,
                "min": minimum depthbin of the final data product,
                "max": maximum depthbin of the final data product
            },
            "energy_deposit": {
                #see "particle_distribution"
                ...
            }
        }
    """
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

