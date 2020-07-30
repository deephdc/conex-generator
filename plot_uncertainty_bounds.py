"""Plot channels of one longitudinal profile with real data uncertainty bounds.

Example: python3 plot_uncertainty_bounds.py 100 200 \
         models/gan/run03/01
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "gsteps",
        type=int,
        help="global label steps - used for bounds"
        )
parser.add_argument(
        "lsteps",
        type=int,
        help="local label steps - used for lines"
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
parser.add_argument(
        "--switch-model",
        action="store_const",
        const=True,
        default=False,
        help="use GAN for bounds and CONEX for line plot"
        )
parser.add_argument(
        "--maxdepth",
        default=1500.0,
        type=float,
        help="maximum depth for plots"
        )
parser.add_argument(
        "--maxplots",
        default=720,
        type=int,
        help="maximum number of plots to be saved"
        )

args = parser.parse_args()
gsteps = args.gsteps
lsteps = args.lsteps
model_base = args.path
file_suffix = args.suffix
switch_model = args.switch_model
maxdepth = args.maxdepth
maxplots = args.maxplots


import numpy as np
import os
import matplotlib

import src


# set paths
root_path = src.utils.get_root_path()
model_path = os.path.join(root_path, model_base)

plot_path = os.path.join(
        src.reports.figures.get_path(),
        model_base,
        "uncertainty_bounds",
        src.utils.timestamp() + file_suffix + "_bounds")
os.makedirs(plot_path)

# load data
gdata = np.load(os.path.join(model_path, "gdata" + file_suffix + ".npy"))
rdata = np.load(os.path.join(model_path, "rdata" + file_suffix + ".npy"))
label = np.load(os.path.join(model_path, "label" + file_suffix + ".npy"))
depth = np.arange(rdata.shape[1])*10.0 + 10.0

# remove > 1500 g/cm^2
dind = np.where(depth <= maxdepth)[0]
gdata = gdata[:,dind,:]
rdata = rdata[:,dind,:]
depth = depth[dind]

# plot models
if switch_model:
    src.plot.uncertainty_bounds(
            plot_path,
            depth,
            gdata, gsteps, "GAN",
            rdata, range(0,gsteps,lsteps), "CONEX",
            label,
            maxplots=maxplots)
else:
    src.plot.uncertainty_bounds(
            plot_path,
            depth,
            rdata, gsteps, "CONEX",
            gdata, range(0,gsteps,lsteps), "GAN",
            label,
            maxplots=maxplots)

