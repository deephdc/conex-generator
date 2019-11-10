import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "bins",
        type=int,
        help="number of bins for combined histograms"
        )
parser.add_argument(
        "path",
        type=str,
        help="model data path relative to the project root directory"
        )
parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="data file suffix"
        )

args = parser.parse_args()
binnum = args.bins
model_base = args.path
file_suffix = args.suffix


import numpy as np
import os

import matplotlib
matplotlib.rcParams.update({"font.size": 20})

import src


root_path = src.utils.get_root_path()
model_path = os.path.join(root_path, model_base)
gaiser_hillas_path = os.path.join(model_path, "gaisser_hillas")

plot_path = os.path.join(
        gaiser_hillas_path,
        "plots",
        src.utils.timestamp() + file_suffix + "_fitparam")
os.makedirs(plot_path)

gdata = np.load(os.path.join(gaiser_hillas_path, "gdata" + file_suffix + "_cond.npy"))
rdata = np.load(os.path.join(gaiser_hillas_path, "rdata" + file_suffix + "_cond.npy"))
label = np.load(os.path.join(gaiser_hillas_path, "label" + file_suffix + "_cond.npy"))

gparam = np.load(os.path.join(gaiser_hillas_path, "gfitparam" + file_suffix + ".npy"))
rparam = np.load(os.path.join(gaiser_hillas_path, "rfitparam" + file_suffix + ".npy"))

condition_nan = np.logical_or(np.any(np.isnan(gparam), axis=(1,2)), np.any(np.isnan(rparam), axis=(1,2)))
index_nan = np.where(condition_nan)[0]
if len(index_nan) != 0:
    print("removing nans from index", index_nan.tolist())
    index_nonan = np.where(np.logical_not(condition_nan))[0]
    gdata = gdata[index_nonan,:,:]
    rdata = rdata[index_nonan,:,:]
    label = label[index_nonan,:]
    gparam = gparam[index_nonan,:,:]
    rparam = rparam[index_nonan,:,:]


# all label
src.plot.gaisser_hillas_hist(plot_path, gparam, rparam, binnum, "all")

# per primary
allprimaries = set(label[:,0].tolist())
for primary in allprimaries:
    index = np.where(label[:,0] == primary)[0]
    tgparam = gparam[index,:,:]
    trparam = rparam[index,:,:]
    tbinnum = max([min([int(np.ceil(len(tgparam)/50)), binnum]), int(np.ceil(binnum/len(allprimaries)))])

    src.plot.gaisser_hillas_hist(plot_path, tgparam, trparam, tbinnum, int(primary))

