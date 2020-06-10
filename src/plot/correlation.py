import matplotlib
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import os

import src.data
import src.utils
log = src.utils.getLogger(__name__)


fontsize = matplotlib.rcParams["font.size"]


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
        "xmax": r"Xmax $(\mathrm{g}\,\mathrm{cm}^{-2})$",
        "x0": r"X0 $(\mathrm{g}\,\mathrm{cm}^{-2})$",
        "lam": "lam $(\mathrm{g}\,\mathrm{cm}^{-2})$",
}


def correlation(plot_path, gparam, rparam, param_name, primary, filetype="png",
                solo=True, solo_bins=30, solo_filetype="svg"):
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

    param_name = param_name.split("_")
    for name in param_name:
        if name not in list(fitparam_layout.keys()):
            msg = f"parameter {name} is not valid"
            log.error(msg)
            raise AssertionError(msg)
    if len(param_name) == 1:
        param_name = param_name*2
    param_index = [fitparam_layout[name] for name in param_name]

    numdata = gparam.shape[0]
    if numdata == 0:
        msg = "input is empty"
        log.error(msg)
        raise AssertionError(msg)

    numchannel = gparam.shape[1]
    numparam = gparam.shape[2]

    # create folder
    foldername = os.path.join("_".join(param_name), primary_name)
    os.makedirs(os.path.join(plot_path, foldername), exist_ok=True)

    # create matrix figures
    matrix_both = plt.figure(1, figsize=(16*numchannel,9*numchannel))
    matrix_solo = plt.figure(2, figsize=(16*numchannel,9*numchannel))

    # plot
    for ii in range(numchannel-1, -1, -1):
        for jj in range(ii, -1, -1):
            channel_name1 = channel_index_to_name[ii]
            channel_name2 = channel_index_to_name[jj]
            
            # get data
            glist1 = gparam[:,ii,param_index[0]]
            glist2 = gparam[:,jj,param_index[1]]
            rlist1 = rparam[:,ii,param_index[0]]
            rlist2 = rparam[:,jj,param_index[1]]
            if "nmax" == param_name[0]:
                glist1 = np.log10(glist1)
                rlist1 = np.log10(rlist1)
            if "nmax" == param_name[1]:
                glist2 = np.log10(glist2)
                rlist2 = np.log10(rlist2)


            # plot both
            filename = "b_" + str(jj) + "_" + str(ii) + "_" \
                       + channel_name2 + "_" + channel_name1 + "_" \
                       + ("_".join(reversed(param_name))) + "_" \
                       + primary_name + "." + filetype

            label1 = "CONEX"
            color1 = "C0"
            label2 = "GAN"
            color2 = "C3"

            title = pretty_name(
                    "correlation: " \
                    + ("/".join(reversed(param_name))) \
                    + ", channel: " \
                    + channel_name2 + "/" + channel_name1 \
                    + ", primary: " \
                    + primary_name)

            xlabel = pretty_name(channel_name1 + ": " + allxlabel[param_name[0]])
            ylabel = pretty_name(channel_name2 + ": " + allxlabel[param_name[1]])

            fig = plt.figure(figsize=(16,9))

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)

            plt.scatter(
                    rlist1, rlist2,
                    label=label1,
                    color=color1,
                    alpha=0.9,
                    marker="o",)

            plt.scatter(
                    glist1, glist2,
                    label=label2,
                    color=color2,
                    alpha=0.8,
                    marker="o",)

            plt.legend()

            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()

            plt.savefig(os.path.join(plot_path, foldername, filename))
            plt.close(fig)

            # fill matrix plot
            if ii != jj:
                plt.figure(1)
                plt.subplot(numchannel, numchannel, numchannel*ii + jj + 1)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                #plt.title(title)
                plt.grid(True)
                plt.scatter(
                        rlist1, rlist2,
                        label=label1,
                        color=color1,
                        alpha=0.9,
                        marker="o",)
                plt.scatter(
                        glist1, glist2,
                        label=label2,
                        color=color2,
                        alpha=0.8,
                        marker="o",)
                plt.legend()


            # plot difference
            filename = "d_" + str(jj) + "_" + str(ii) + "_" \
                       + channel_name2 + "_" + channel_name1 + "_" \
                       + ("_".join(reversed(param_name))) + "_" \
                       + primary_name + "." + solo_filetype

            label = "CONEX - GAN"
            color = "C7"

            title = pretty_name(
                    "correlation: " \
                    + ("/".join(reversed(param_name))) \
                    + ", channel: " \
                    + channel_name2 + "/" + channel_name1 \
                    + ", primary: " \
                    + primary_name)

            xlabel = pretty_name(channel_name1 + ": " + allxlabel[param_name[0]])
            ylabel = pretty_name(channel_name2 + ": " + allxlabel[param_name[1]])

            hr, xedges, yedges = np.histogram2d(rlist1, rlist2,
                                                solo_bins,
                                                range=[xlim, ylim],)
            hg, xedges, yedges = np.histogram2d(glist1, glist2,
                                                solo_bins,
                                                range=[xlim, ylim],)
            diff = hr - hg

            fig = plt.figure(figsize=(16,9))

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)

            plt.pcolormesh(xedges, yedges, np.abs(diff.T))

            plt.colorbar()
            #plt.legend()
            plt.savefig(os.path.join(plot_path, foldername, filename))
            plt.close(fig)

            # fill matrix plot
            if ii != jj:
                plt.figure(1)
                plt.subplot(numchannel, numchannel, numchannel*jj + ii + 1)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                #plt.title(title)
                plt.grid(True)
                plt.pcolormesh(xedges, yedges, np.abs(diff.T))
                plt.colorbar()
                #plt.legend()


            if solo:
                # plot solo r
                filename = "r_" + str(jj) + "_" + str(ii) + "_" \
                           + channel_name2 + "_" + channel_name1 + "_" \
                           + ("_".join(reversed(param_name))) + "_" \
                           + primary_name + "." + solo_filetype

                label = "CONEX"
                color = "C0"

                title = pretty_name(
                        "correlation: " \
                        + ("/".join(reversed(param_name))) \
                        + ", channel: " \
                        + channel_name2 + "/" + channel_name1 \
                        + ", primary: " \
                        + primary_name)

                xlabel = pretty_name(channel_name1 + ": " + allxlabel[param_name[0]])
                ylabel = pretty_name(channel_name2 + ": " + allxlabel[param_name[1]])

                fig = plt.figure(figsize=(16,9))

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.grid(True)

                plt.hist2d(
                        rlist1, rlist2,
                        solo_bins,
                        range=[xlim, ylim],
                        label=label,)

                plt.colorbar()
                #plt.legend()
                plt.savefig(os.path.join(plot_path, foldername, filename))
                plt.close(fig)

                # fill matrix plot
                if ii != jj:
                    plt.figure(2)
                    plt.subplot(numchannel, numchannel, numchannel*ii + jj + 1)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    #plt.title(title)
                    plt.grid(True)
                    plt.hist2d(
                            rlist1, rlist2,
                            solo_bins,
                            range=[xlim, ylim],
                            label=label,)
                    plt.colorbar()
                    #plt.legend()


                # plot solo g
                filename = "g_" + str(jj) + "_" + str(ii) + "_" \
                           + channel_name2 + "_" + channel_name1 + "_" \
                           + ("_".join(reversed(param_name))) + "_" \
                           + primary_name + "." + solo_filetype

                label = "GAN"
                color = "C3"

                title = pretty_name(
                        "correlation: " \
                        + ("/".join(reversed(param_name))) \
                        + ", channel: " \
                        + channel_name2 + "/" + channel_name1 \
                        + ", primary: " \
                        + primary_name)

                xlabel = pretty_name(channel_name1 + ": " + allxlabel[param_name[0]])
                ylabel = pretty_name(channel_name2 + ": " + allxlabel[param_name[1]])

                fig = plt.figure(figsize=(16,9))

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)
                plt.grid(True)

                plt.hist2d(
                        glist1, glist2,
                        solo_bins,
                        range=[xlim, ylim],
                        label=label,)

                plt.colorbar()
                #plt.legend()
                plt.savefig(os.path.join(plot_path, foldername, filename))
                plt.close(fig)

                # fill matrix plot
                if ii != jj:
                    plt.figure(2)
                    plt.subplot(numchannel, numchannel, numchannel*jj + ii + 1)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    #plt.title(title)
                    plt.grid(True)
                    plt.hist2d(
                            glist1, glist2,
                            solo_bins,
                            range=[xlim, ylim],
                            label=label,)
                    plt.colorbar()
                    #plt.legend()

    
    # save matrix plots
    filename = "matrix_both." + filetype
    title = pretty_name(
            "correlation: " \
            + ("/".join(reversed(param_name))) \
            + ", primary: " \
            + primary_name) \
            + ", down/left: scatter, top/right: difference"
    fig = plt.figure(1)
    plt.suptitle(title, fontweight="bold", fontsize=fontsize*(numchannel-2),
                 horizontalalignment="center", verticalalignment="top")
    plt.savefig(os.path.join(plot_path, foldername, filename))
    plt.close(fig)

    filename = "matrix_solo." + solo_filetype
    title = pretty_name(
            "correlation: " \
            + ("/".join(reversed(param_name))) \
            + ", primary: " \
            + primary_name) \
            + ", down/left: CONEX, top/right: GAN"
    fig = plt.figure(2)
    plt.suptitle(title, fontweight="bold", fontsize=fontsize*(numchannel-2),
                 horizontalalignment="center", verticalalignment="top")
    plt.savefig(os.path.join(plot_path, foldername, filename))
    plt.close(fig)

