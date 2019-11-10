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
import scipy as sp
import scipy.stats as spstats
import json
import os
import itertools as it
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 20})

import src

root_path = src.utils.get_root_path()
model_path = os.path.join(root_path, model_base)
gaiser_hillas_path = os.path.join(model_path, "gaisser_hillas")

plot_path = os.path.join(model_path, "plots", src.utils.timestamp() + file_suffix)
os.makedirs(plot_path)

def pretty_name(string : str):
    string = string.replace("mup", "muon+")
    string = string.replace("mum", "muon-")
    string = string.replace("nmax", "Nmax")
    string = string.replace("xmax", "Xmax")
    string = string.replace("x0", "X0")
    string = string.replace("lam", "lambda")
    return string

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

with open(os.path.join(gaiser_hillas_path, "fitparam" + file_suffix + "_metadata.json"), "r") as fp:
    fpmeta = json.load(fp)
fp_index_to_name = {value: key for key,value in fpmeta["fitparam_layout"].items()}

data_layout = src.data.numpy_data_layout()["particle_distribution"]
channel_index_to_name = {value: key for key, value in data_layout.items()}

particle_label = src.data.numpy_particle_label()
primary_index_to_name = {value: key for key, value in particle_label.items()}

numdata = gparam.shape[0]
numchannel = gparam.shape[1]
numparam = gparam.shape[2]

# param stuff
allxlabel = {
        "nmax": "log(Nmax)",
        "xmax": "Xmax (g/cm^2)",
        "x0": "X0 (g/cm^2)",
        "lam": "lam (g/cm^2)",
}

def plot_distributions(gparam, rparam, numbins, primary_name, solo=False, outlier=True):
    for pind, cind in it.product(range(numparam), range(numchannel)):
        param_name = fp_index_to_name[pind]
        channel_name = channel_index_to_name[cind]

        glist = gparam[:,cind,pind]
        rlist = rparam[:,cind,pind]
        if param_name == "nmax":
            glist = np.log10(glist)
            rlist = np.log10(rlist)

        # make bins
        glower = np.min(glist)
        gupper = np.max(glist)
        rlower = np.min(rlist)
        rupper = np.max(rlist)
        rrange = rupper - rlower
        rwidth = rrange/numbins

        lower = max([min([glower, rlower]), rlower-0.25*rrange])
        upper = min([max([gupper, rupper]), rupper+0.25*rrange])
        numbins_new = int(np.ceil((upper - lower) / rwidth))
        bwidth = (upper - lower) / numbins_new

        bins = np.linspace(lower-bwidth, upper+bwidth, numbins_new+1+2)
        
        # plot both
        filename = "b_" + channel_name + "_" + param_name + "_" + primary_name + ".svg"
        label1 = "CONEX"
        color1 = "C0"
        label2 = "GAN"
        color2 = "C3"

        fig = plt.figure(figsize=(16,9))
        title = pretty_name(
                "distribution: " \
                + param_name \
                + ", channel: " \
                + channel_name \
                + ", primary: " \
                + primary_name)
        xlabel = pretty_name(allxlabel[param_name])

        plt.xlabel(xlabel)
        plt.ylabel("counts")
        plt.title(title)

        if outlier:
            plt.hist(
                    np.concatenate((bins[0]*np.ones(np.sum(rlist < bins[1])), bins[-1]*np.ones(np.sum(rlist > bins[-2])))),
                    bins=bins,
                    color=color1,
                    alpha=0.9,
                    hatch="/",
                    edgecolor="black",
                    linewidth=2.0)

            plt.hist(
                    np.concatenate((bins[0]*np.ones(np.sum(glist < bins[1])), bins[-1]*np.ones(np.sum(glist > bins[-2])))),
                    bins=bins,
                    color=color2,
                    alpha=0.8,
                    hatch="/",
                    edgecolor="black",
                    linewidth=2.0,
                    rwidth=0.65)

        plt.hist(
                rlist,
                bins=bins[1:-1],
                label=label1,
                color=color1,
                alpha=0.9,
                edgecolor="black",
                linewidth=2.0)

        plt.hist(
                glist,
                bins=bins[1:-1],
                label=label2,
                color=color2,
                alpha=0.8,
                edgecolor="black",
                linewidth=2.0,
                rwidth=0.65)

        plt.legend()
        plt.savefig(os.path.join(plot_path, filename))
        plt.close(fig)


        if solo:
            # plot solo g
            filename = "g_" + channel_name + "_" + param_name + "_" + primary_name + ".svg"
            label = "GAN"
            color = "C3"
            clist = glist

            fig = plt.figure(figsize=(16,9))
            title = pretty_name(
                    "distribution: " \
                    + param_name \
                    + ", channel: " \
                    + channel_name \
                    + ", primary: " \
                    + primary_name)
            xlabel = pretty_name(allxlabel[param_name])

            plt.xlabel(xlabel)
            plt.ylabel("counts")
            plt.title(title)

            if outlier:
                plt.hist(
                        np.concatenate((bins[0]*np.ones(np.sum(clist < bins[1])), bins[-1]*np.ones(np.sum(clist > bins[-2])))),
                        bins=bins,
                        color=color,
                        alpha=1.0,
                        hatch="/",
                        edgecolor="black",
                        linewidth=2.0)

            plt.hist(
                    clist,
                    bins=bins[1:-1],
                    label=label,
                    color=color,
                    alpha=1.0,
                    edgecolor="black",
                    linewidth=2.0)

            plt.legend()
            plt.savefig(os.path.join(plot_path, filename))
            plt.close(fig)


            # plot solo r
            filename = "r_" + channel_name + "_" + param_name + "_" + primary_name + ".svg"
            label = "CONEX"
            color = "C0"
            clist = rlist

            fig = plt.figure(figsize=(16,9))
            title = pretty_name(
                    "distribution: " \
                    + param_name \
                    + ", channel: " \
                    + channel_name \
                    + ", primary: " \
                    + primary_name)
            xlabel = pretty_name(allxlabel[param_name])

            plt.xlabel(xlabel)
            plt.ylabel("counts")
            plt.title(title)

            if outlier:
                plt.hist(
                        np.concatenate((bins[0]*np.ones(np.sum(clist < bins[1])), bins[-1]*np.ones(np.sum(clist > bins[-2])))),
                        bins=bins,
                        color=color,
                        alpha=1.0,
                        hatch="/",
                        edgecolor="black",
                        linewidth=2.0)

            plt.hist(
                    clist,
                    bins=bins[1:-1],
                    label=label,
                    color=color,
                    alpha=1.0,
                    edgecolor="black",
                    linewidth=2.0)

            plt.legend()
            plt.savefig(os.path.join(plot_path, filename))
            plt.close(fig)


# all label
plot_distributions(gparam, rparam, binnum, "all")

# per primary
allprimaries = set(label[:,0].tolist())
for primary in allprimaries:
    index = np.where(label[:,0] == primary)[0]
    tgparam = gparam[index,:,:]
    trparam = rparam[index,:,:]
    tbinnum = max([min([int(np.ceil(len(tgparam)/50)), binnum]), int(np.ceil(binnum/len(allprimaries)))])
    primary_name = primary_index_to_name[int(primary)]

    plot_distributions(tgparam, trparam, tbinnum, primary_name)

