import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "path",
        type=str,
        help="model data path relative to the project root directory"
        )
parser.add_argument(
        "bins",
        type=int,
        help="number of bins for combined histograms"
        )
args = parser.parse_args()
model_base = args.path
binnum = args.bins

import numpy as np
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

plot_path = os.path.join(model_path, "plots", src.utils.timestamp())
os.makedirs(plot_path)

def pretty_name(string : str):
    string = string.replace("mup", "muon+")
    string = string.replace("mum", "muon-")
    string = string.replace("nmax", "Nmax")
    string = string.replace("xmax", "Xmax")
    string = string.replace("x0", "X0")
    string = string.replace("lam", "lambda")
    return string

gdata = np.load(os.path.join(gaiser_hillas_path, "gdata_cond.npy"))
rdata = np.load(os.path.join(gaiser_hillas_path, "rdata_cond.npy"))
label = np.load(os.path.join(gaiser_hillas_path, "label_cond.npy"))

gparam = np.load(os.path.join(gaiser_hillas_path, "gfitparam.npy"))
rparam = np.load(os.path.join(gaiser_hillas_path, "rfitparam.npy"))

with open(os.path.join(gaiser_hillas_path, "fitparam_metadata.json"), "r") as fp:
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


def plot_distributions(gparam, rparam, numbins, primary_name):
    for pind, cind in it.product(range(numparam), range(numchannel)):
        param_name = fp_index_to_name[pind]
        channel_name = channel_index_to_name[cind]

        glist = gparam[:,cind,pind]
        rlist = rparam[:,cind,pind]
        if param_name == "nmax":
            glist = np.log10(glist)
            rlist = np.log10(rlist)
        
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

        _, bins, _ = plt.hist(
                rlist,
                bins=numbins,
                label=label1,
                color=color1,
                alpha=0.9,
                edgecolor="black",
                linewidth=2.0)

        plt.hist(
                glist,
                bins=bins,
                label=label2,
                color=color2,
                alpha=0.8,
                edgecolor="black",
                linewidth=2.0,
                rwidth=0.65)

        plt.legend()
        plt.savefig(os.path.join(plot_path, filename))
        plt.close(fig)


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

        plt.hist(
                clist,
                bins=bins,
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

        plt.hist(
                clist,
                bins=bins,
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

allprimaries = set(label[:,0].tolist())
for primary in allprimaries:
    index = np.where(label[:,0] == primary)[0]
    tgparam = gparam[index,:,:]
    trparam = rparam[index,:,:]
    primary_name = primary_index_to_name[int(primary)]
    plot_distributions(tgparam, trparam, int(binnum/len(allprimaries))+1, primary_name)

