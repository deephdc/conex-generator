import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import datetime

timestamp = str(datetime.datetime.now()).replace(" ","-").replace(":","-")
plotdir = os.path.join(".", "plots", timestamp)
os.mkdir(plotdir)

matplotlib.rcParams.update({"font.size": 20})
#matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['text.latex.unicode']=True

data_path = "."

gdata_path = os.path.join(data_path, "gdata.npy")
label_path = os.path.join(data_path, "label.npy")
rdata_path = os.path.join(data_path, "rdata.npy")

gdata = np.load("gdata.npy")
label = np.load("label.npy")
rdata = np.load("rdata.npy")

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

particle_list = {
        0: "gamma",
        1: "electron",
        2: "proton",
        3: "helium",
        4: "oxygen",
        5: "iron",
}

data_list = {
        0: "gamma",
        1: "positron",
        2: "electron",
        3: "muon+",
        4: "muon-",
        5: "hadron",
        6: "charged",
        7: "nuclei",
}

depth = np.array(range(gdata.shape[1]))*10.0
avgwindowsize = 5
avgdepth = running_mean(depth, avgwindowsize)

for ii in range(0,500,10):
    print(ii)

    rdata_avg = np.mean(rdata[ii:ii+10], axis=0)
    rdata_std = np.std(rdata[ii:ii+10], axis=0)

    inparticle = particle_list[int(label[ii,0])]
    energy = label[ii,1] / 1.0e9
    theta = label[ii,2]
    phi = label[ii,3]

    for jj in range(8):
        outparticle = data_list[jj]

        rdata_avg_rm = running_mean(rdata_avg[:,jj], avgwindowsize)
        rdata_std_rm = running_mean(rdata_std[:,jj], avgwindowsize)

        for kk in range(10):
            fig = plt.figure(figsize=(16,9))


            plt.xlabel("depth (g/cm^2)")
            plt.ylabel("particle number (" + outparticle + ")")

            plt.grid(True)

            titlestring = inparticle + ", E = " + str(round(energy,2)) + "E18 eV, theta = " + \
                          str(round(theta,1)) + " deg, phi = " + str(round(phi,1)) + " deg"
            plt.title(titlestring)

            plt.fill_between(
                    avgdepth,
                    rdata_avg_rm+rdata_std_rm,
                    rdata_avg_rm-rdata_std_rm,
                    color="C0", alpha=0.3, label="CONEX 1 sigma")

            plt.fill_between(
                    avgdepth,
                    rdata_avg_rm+2*rdata_std_rm,
                    rdata_avg_rm+rdata_std_rm,
                    color="C2", alpha=0.3, label="CONEX 2 sigma")

            plt.fill_between(
                    avgdepth,
                    rdata_avg_rm-2*rdata_std_rm,
                    rdata_avg_rm-rdata_std_rm,
                    color="C2", alpha=0.3)

            plt.fill_between(
                    avgdepth,
                    rdata_avg_rm+3*rdata_std_rm,
                    rdata_avg_rm+2*rdata_std_rm,
                    color="C1", alpha=0.3, label="CONEX 3 sigma")

            plt.fill_between(
                    avgdepth,
                    rdata_avg_rm-3*rdata_std_rm,
                    rdata_avg_rm-2*rdata_std_rm,
                    color="C1", alpha=0.3)

            temp = running_mean(gdata[ii+kk,:,jj], avgwindowsize)
            plt.plot(avgdepth, temp, linewidth=5, label="GAN", color="C3")

            plt.legend()
            
            filename = str(ii+kk) + "_" + \
                       titlestring.replace(" ","").replace(",","_") + \
                       "_" + outparticle + ".png" 
            plt.savefig(os.path.join(plotdir, filename))
            plt.close(fig)

