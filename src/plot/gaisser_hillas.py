import matplotlib
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import os

import src.data
import src.utils
log = src.utils.getLogger(__name__)


def pretty_name(string : str):
    string = string.replace("mup", "muon+")
    string = string.replace("mum", "muon-")
    string = string.replace("nmax", "Nmax")
    string = string.replace("xmax", "Xmax")
    string = string.replace("x0", "X0")
    string = string.replace("lam", "lambda")
    return string


data_layout = src.data.numpy_data_layout()["particle_distribution"]
channel_index_to_name = {value: key for key, value in data_layout.items()}

particle_label = src.data.numpy_particle_label()
primary_index_to_name = {value: key for key, value in particle_label.items()}

fitparam_layout = {
        "nmax": 0,
        "xmax": 1,
        "x0": 2,
        "lam": 3,
}
fp_index_to_name = {value: key for key,value in fitparam_layout.items()}

allxlabel = {
        "nmax": "log(Nmax)",
        "xmax": "Xmax (g/cm^2)",
        "x0": "X0 (g/cm^2)",
        "lam": "lam (g/cm^2)",
}


def gaisser_hillas_hist(plot_path, gparam, rparam, binnum, primary, solo=False, outlier=True):
    if len(gparam) != len(rparam):
        msg = "gparam and rparam must have same length"
        log.error(msg)
        raise AssertionError(msg)

    if isinstance(primary, int):
        primary_name = primary_index_to_name[primary]
    else:
        if isinstance(primary, str):
            primary_name = primary
        else:
            msg = f"primary {primary} is neither int nor string"
            log.error(msg)
            raise AssertionError(msg)

    numdata = gparam.shape[0]
    if numdata == 0:
        msg = "input is empty"
        log.error(msg)
        raise AssertionError(msg)


    numchannel = gparam.shape[1]
    numparam = gparam.shape[2]

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
        rwidth = rrange/binnum

        lower = max([min([glower, rlower]), rlower-0.25*rrange])
        upper = min([max([gupper, rupper]), rupper+0.25*rrange])
        binnum_new = int(np.ceil((upper - lower) / rwidth))
        bwidth = (upper - lower) / binnum_new

        bins = np.linspace(lower-bwidth, upper+bwidth, binnum_new+1+2)
        
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

