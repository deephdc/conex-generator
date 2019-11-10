import numpy as np
import matplotlib.pyplot as plt
import os

import src.data
import src.utils
log = src.utils.getLogger(__name__)


data_layout = src.data.numpy_data_layout()["particle_distribution"]
channel_index_to_name = {value: key for key, value in data_layout.items()}

particle_label = src.data.numpy_particle_label()
primary_index_to_name = {value: key for key, value in particle_label.items()}


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def fill_uncertainity_bounds(depth, data, plot_label, avgwindowsize = 5):
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

    depth_rm = running_mean(depth, avgwindowsize)
    data_avg_rm = running_mean(data_avg, avgwindowsize)
    data_std_rm = running_mean(data_std, avgwindowsize)

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


def plot_line(depth, data, plot_label, avgwindowsize = 5):
    numpoints = data.shape[0]
    if numpoints != len(depth):
        msg = "depth has different length than data"
        log.error(msg)
        raise AssertionError(msg)

    depth_rm = running_mean(depth, avgwindowsize)
    data_rm = running_mean(data, avgwindowsize)

    plt.plot(depth_rm, data_rm, linewidth=5, label=plot_label, color="C3")


def uncertainty_bounds(
        plot_path,
        depth,
        bdata, bsteps, bplot_label,
        ldata, lsteps, lplot_label,
        label,
        avgwindowsize = 5):

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


    for ii in range(0,numdata,bsteps):
        for jj in range(0,bsteps,lsteps):
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
                plt.ylabel("particle number (" + outparticle + ")")
                plt.title(titlestring)
                plt.grid(True)

                tbdata = bdata[ii:ii+bsteps,:,kk]
                fill_uncertainity_bounds(depth, tbdata, bplot_label, avgwindowsize)

                tldata = ldata[ii+jj,:,kk]
                plot_line(depth, tldata, lplot_label, avgwindowsize)
                
                plt.legend()

                plt.savefig(os.path.join(plot_path, filename))
                plt.close(fig)

