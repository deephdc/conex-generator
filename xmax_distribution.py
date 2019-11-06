import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

import src

model_base = "models/gan/run01/02"
root_path = src.utils.get_root_path()
model_path = os.path.join(root_path, model_base)
plot_path = os.path.join(model_path, "plots", src.utils.timestamp())
#os.mkdir(plot_path)

gdata = np.load(os.path.join(model_path, "gdata.npy"))
rdata = np.load(os.path.join(model_path, "rdata.npy"))
label = np.load(os.path.join(model_path, "label.npy"))

# filter data
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

depth = np.arange(rdata.shape[1])*10.0 + 10.0

params = src.analysis.xmax.gaisser_hillas_fit(depth, rdata[0,:,0])

