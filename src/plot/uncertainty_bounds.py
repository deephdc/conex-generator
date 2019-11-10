import numpy as np
import matplotlib.pyplot as plt
import os

import src.data
import src.analysis
import src.utils
log = src.utils.getLogger(__name__)


def pretty_name(string : str):
    string = string.replace("mup", "muon+")
    string = string.replace("mum", "muon-")
    return string


data_layout = src.data.numpy_data_layout()["particle_distribution"]
channel_index_to_name = {value: key for key, value in data_layout.items()}

particle_label = src.data.numpy_particle_label()
primary_index_to_name = {value: key for key, value in particle_label.items()}


def fill_bounds(depth, data, plot_label, avgwindowsize = 1):
    numdata = data.shape[0]
    if numdata == 0:
        msg = "input is empty"
        log.error(msg)
        raise AssertionError(msg)

    numpoints = data.shape[1]
    if numpoints != len(depth):
        msg = "depth has different length than data"
        log.error(msg)
        raise AssertionError(msg)

    data_avg = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    depth_rm = src.analysis.running_mean(depth, avgwindowsize)
    data_avg_rm = src.analysis.running_mean(data_avg, avgwindowsize)
    data_std_rm = src.analysis.running_mean(data_std, avgwindowsize)

    plt.fill_between(
            depth_rm,
            data_avg_rm+1*data_std_rm,
            data_avg_rm-1*data_std_rm,
            color="C0", alpha=0.3, label=plot_label + " ±1 std")

    plt.fill_between(
            depth_rm,
            data_avg_rm+2*data_std_rm,
            data_avg_rm+1*data_std_rm,
            color="C2", alpha=0.3, label=plot_label + " ±2 std")

    plt.fill_between(
            depth_rm,
            data_avg_rm-2*data_std_rm,
            data_avg_rm-1*data_std_rm,
            color="C2", alpha=0.3)

    plt.fill_between(
            depth_rm,
            data_avg_rm+3*data_std_rm,
            data_avg_rm+2*data_std_rm,
            color="C1", alpha=0.3, label=plot_label + " ±3 std")

    plt.fill_between(
            depth_rm,
            data_avg_rm-3*data_std_rm,
            data_avg_rm-2*data_std_rm,
            color="C1", alpha=0.3)


def red_line(depth, data, plot_label, avgwindowsize = 1):
    numpoints = data.shape[0]
    if numpoints != len(depth):
        msg = "depth has different length than data"
        log.error(msg)
        raise AssertionError(msg)

    depth_rm = src.analysis.running_mean(depth, avgwindowsize)
    data_rm = src.analysis.running_mean(data, avgwindowsize)

    plt.plot(depth_rm, data_rm, linewidth=5, label=plot_label, color="C3")


def uncertainty_bounds(
        plot_path,
        depth,
        bdata, bsteps, bplot_label,
        ldata, lindex, lplot_label,
        label,
        avgwindowsize = 5,
        maxplots = 720):

    numdata = bdata.shape[0]
    if numdata == 0:
        msg = "input is empty"
        log.error(msg)
        raise AssertionError(msg)

    if len(bdata) != len(ldata) or len(bdata) != len(label):
        msg = "numdata do not match"
        log.error(msg)
        raise AssertionError(msg)

    numchannel = bdata.shape[2]
    if numchannel != ldata.shape[2]:
        msg = "numchannel do not match"
        log.error(msg)
        raise AssertionError(msg)

    try:
        lsteps = iter(lindex)
        for ind in lsteps:
            if ind > bsteps - 1 or ind < -bsteps:
                msg = "line index out of bounds"
                log.error(msg)
                raise AssertionError(msg)
    except:
        log.error("lindex error")
        raise

    curplot = 0
    for ii in range(0,numdata,bsteps):
        lsteps = iter(lindex)

        for jj in lsteps:
            if jj < 0:
                jj = bsteps - jj

            for kk in range(numchannel):
                outparticle = channel_index_to_name[kk]
                inparticle = primary_index_to_name[int(label[ii+jj,0])]
                energy = label[ii+jj,1] / 1e9
                theta = label[ii+jj,2]
                phi = label[ii+jj,3]

                titlestring = inparticle \
                        + ", E = " + str(round(energy,2)) \
                        + "E18 eV, theta = " + str(round(theta,1)) \
                        + " deg, phi = " + str(round(phi,1)) \
                        + " deg"

                filename = str(ii+jj) + "_" \
                        + titlestring.replace(" ","").replace(",","_") \
                        + "_" + outparticle + ".svg"

                fig = plt.figure(figsize=(16,9))

                plt.xlabel("depth (g/cm^2)")
                plt.ylabel(pretty_name("particle number (" + outparticle + ")"))
                plt.title(titlestring)
                plt.grid(True)

                tbdata = bdata[ii:ii+bsteps,:,kk]
                fill_bounds(depth, tbdata, bplot_label, avgwindowsize)

                tldata = ldata[ii+jj,:,kk]
                red_line(depth, tldata, lplot_label, avgwindowsize)
                
                plt.legend()

                plt.savefig(os.path.join(plot_path, filename))
                plt.close(fig)

                curplot += 1
                if curplot >= maxplots:
                    return

