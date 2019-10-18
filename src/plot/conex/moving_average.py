import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams.update({"font.size": 20})

data = np.load("/cr/users/koepke/data/conex/data.npy")
label = np.load("/cr/users/koepke/data/conex/label.npy")

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


for kk in range(500):
    print(kk)
    for jj in range(8):
        fig = plt.figure(figsize=(16,9))
        outparticle = data_list[jj]
        plt.xlabel("depth (g/cm^2)")
        plt.ylabel("particle number (" + outparticle + ")")

        plt.grid(True)

        for ii in range(kk*10,(kk+1)*10):
            inparticle = particle_list[int(label[ii,0])]
            energy = label[ii,1] / 1.0e9
            theta = label[ii,2]
            phi = label[ii,3]
            titlestring = inparticle + ", E = " + str(round(energy,2)) + "E18 eV, theta = " + \
                          str(round(theta,1)) + " deg, phi = " + str(round(phi,1)) + " deg"
            plt.title(titlestring)
            temp = running_mean(data[ii,:,jj], avgwindowsize)
            plt.plot(avgdepth, temp)

        plt.savefig(str(kk) + "_" + \
                    titlestring.replace(" ","").replace(",","_") + \
                    "_" + outparticle + ".png")
        plt.close(fig)

