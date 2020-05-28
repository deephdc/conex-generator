import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "path",
        type=str,
        help="path to Gaisser-Hillas fit parameter (relative to the project root directory)"
        )

#args = parser.parse_args()
#model_base = args.path
model_base = "models/talk/run03/03/gaisser_hillas_ecut_1.00e+08_1.00e+10_tcut_3.50e+01_7.00e+01_fit-error_True"


import numpy as np
import os
import matplotlib

import src


# paths
root_path = src.utils.get_root_path()
gaiser_hillas_path = os.path.join(root_path, model_base)

# load data
gdata = np.load(os.path.join(gaiser_hillas_path, "gdata_cond.npy"))
rdata = np.load(os.path.join(gaiser_hillas_path, "rdata_cond.npy"))
label = np.load(os.path.join(gaiser_hillas_path, "label_cond.npy"))

gparam = np.load(os.path.join(gaiser_hillas_path, "gfitparam.npy"))
rparam = np.load(os.path.join(gaiser_hillas_path, "rfitparam.npy"))

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

assert gparam.shape == rparam.shape

# plot path
plot_path = os.path.join(
        src.reports.figures.get_path(),
        model_base,
        src.utils.timestamp() + "_correlation")
os.makedirs(plot_path)

# nmax
src.plot.correlation(plot_path, gparam, rparam, "nmax_xmax", "all", solo=True)

