import numpy as np
import scipy as sp
import os
import matplotlib
import matplotlib.pyplot as plt

import src

model_base = "models/gan/run01/02"
root_path = src.utils.get_root_path()
model_path = os.path.join(root_path, model_base)
plot_path = os.path.join(model_path, "plots", src.utils.timestamp())

gdata = np.load(os.path.join(model_path, "gdata.npy"))
rdata = np.load(os.path.join(model_path, "rdata.npy"))
label = np.load(os.path.join(model_path, "label.npy"))

# prepare data
condition1 = label[:,0] >= 2
print("condition1: ", np.sum(condition1))
condition2 = label[:,1] >= 1e9
print("condition2: ", np.sum(condition2))
condition3 = label[:,2] >= 35
print("condition3: ", np.sum(condition3))
condition = np.logical_and(np.logical_and(condition1, condition2), condition3)
print("total condition: ", np.sum(condition))
index = np.where(condition)[0]

gdata = gdata[index,:,:]
rdata = rdata[index,:,:]
label = label[index,:]

def find_rdata_cut(rdata):
    cond = rdata[:,:,:-1] == 0.0
    indlist = np.where(cond)
    index1 = indlist[1]
    return np.min(index1[index1 > 50]) - 5

dcut = find_rdata_cut(rdata)
gdata = gdata[:,0:dcut,:]
rdata = rdata[:,0:dcut,:]

gdata = np.abs(gdata) + 0.01
rdata = np.abs(rdata) + 0.01

numdata = rdata.shape[0]
channel = [0,1,2,3,4,5,6]
depth = np.arange(rdata.shape[1])*10.0 + 10.0

# calculate distributions
np.seterr(all="raise")
sp.special.seterr(all="raise")
gparam = np.zeros((numdata, len(channel), 4))
rparam = np.zeros((numdata, len(channel), 4))

gerr = 0
rerr = 0
for ii in range(numdata):
    for jj in range(len(channel)):
        try:
            gparam[ii,jj,:] = src.analysis.xmax.gaisser_hillas_fit(depth, gdata[ii,:,jj])
        except:
            gparam[ii,jj,:] = np.nan
            gerr += 1
        try:
            rparam[ii,jj,:] = src.analysis.xmax.gaisser_hillas_fit(depth, rdata[ii,:,jj])
        except:
            rparam[ii,jj,:] = np.nan
            rerr += 1
print("gerr", gerr, "rerr", rerr)

gxmax = gparam[:,:,1]
rxmax = rparam[:,:,1]

# plots
os.mkdir(plot_path)

bins = np.linspace(600,1200,21)

for jj in range(len(channel)):
    filename = "gxmax_" + str(jj) + ".png"
    fig = plt.figure()
    plt.hist(gxmax[:,jj], bins=bins)
    plt.savefig(os.path.join(plot_path, filename))
    plt.close(fig)

    filename = "rxmax_" + str(jj) + ".png"
    fig = plt.figure()
    plt.hist(rxmax[:,jj], bins=bins)
    plt.savefig(os.path.join(plot_path, filename))
    plt.close(fig)

