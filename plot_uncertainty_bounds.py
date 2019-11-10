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

args = parser.parse_args()
gsteps = args.gsteps
lsteps = args.lsteps
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

plot_path = os.path.join(model_path, "plots", src.utils.timestamp() + file_suffix)
os.makedirs(plot_path)

gdata = np.load(os.path.join(model_path, "gdata" + file_suffix + ".npy"))
rdata = np.load(os.path.join(model_path, "rdata" + file_suffix + ".npy"))
label = np.load(os.path.join(model_path, "label" + file_suffix + ".npy"))
depth = np.arange(rdata.shape[1])*10.0 + 10.0

src.plot.uncertainty_bounds(
        plot_path,
        depth,
        rdata, gsteps, "CONEX",
        gdata, lsteps, "GAN",
        label)
