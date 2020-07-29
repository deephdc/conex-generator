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
        help="path to Gaisser-Hillas fit parameter (relative to the project root directory)"
        )
parser.add_argument(
        "--logscale",
        action="store_const",
        const=True,
        default=False,
        help="use logarithmic colorscale for density plots"
        )
parser.add_argument(
        "--metrics",
        nargs="+",
        default=["wasserstein"],
        choices=["wasserstein", "chi2"],
        help="list of metrics to calculate"
        )
args = parser.parse_args()

args = parser.parse_args()
binnum = args.bins
model_base = args.path
logscale = args.logscale
metrics = args.metrics


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
analysis_path = os.path.join(
        src.reports.analysis.get_path(),
        model_base,
        src.utils.timestamp() + "_histogram" \
                              + f"_logscale_{logscale}" \
                              + f"_metrics_" + "_".join(metrics)
        )
os.makedirs(analysis_path)

