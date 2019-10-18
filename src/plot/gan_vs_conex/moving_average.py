import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams.update({"font.size": 20})
#matplotlib.rcParams['text.usetex']=True                                         
#matplotlib.rcParams['text.latex.unicode']=True    

data = np.load("gdata.npy")
labels = np.load("glabel.npy")
realdata = np.load("rdata.npy")

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

depth = np.array(range(data.shape[1]))*10.0
avgwindowsize = 5
avgdepth = running_mean(depth, avgwindowsize)

for ii in range(500):
    print(ii)
    for jj in range(8):
        fig = plt.figure(figsize=(16,9))

        inparticle = particle_list[int(labels[ii,0])]
        energy = labels[ii,1] / 1.0e9
        theta = labels[ii,2]
        phi = labels[ii,3]

        outparticle = data_list[jj]
        plt.xlabel("depth (g/cm^2)")
        plt.ylabel("particle number (" + outparticle + ")")

        plt.grid(True)

        titlestring = inparticle + ", E = " + str(round(energy,2)) + "E18 eV, theta = " + \
                      str(round(theta,1)) + " deg, phi = " + str(round(phi,1)) + " deg"
        plt.title(titlestring)

        temp = running_mean(realdata[ii,:,jj], avgwindowsize)
        plt.plot(avgdepth, temp, linewidth=3, label="CONEX")
        temp = running_mean(data[ii,:,jj], avgwindowsize)
        plt.plot(avgdepth, temp, linewidth=3, label="GAN")

        plt.legend()

        plt.savefig(str(ii) + "_" + \
                    titlestring.replace(" ","").replace(",","_") + \
                    "_" + outparticle + ".png")
        plt.close(fig)

