import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "path",
        type=str,
        help="model data path relative to the project root directory"
        )
parser.add_argument(
        "ecut",
        type=float,
        help="energy cut in GeV"
        )
args = parser.parse_args()
model_base = args.path
ecut = args.ecut


import numpy as np
import scipy as sp
import os
import json

import src

root_path = src.utils.get_root_path()
model_path = os.path.join(root_path, model_base)
gaisser_hillas_path = os.path.join(model_path, "gaisser_hillas")
if not os.path.isdir(gaisser_hillas_path):
    os.mkdir(gaisser_hillas_path)

gdata = np.load(os.path.join(model_path, "gdata.npy"))
rdata = np.load(os.path.join(model_path, "rdata.npy"))
label = np.load(os.path.join(model_path, "label.npy"))

# prepare data
condition1 = label[:,0] >= 2
print("condition1: ", np.sum(condition1))
condition2 = label[:,1] >= ecut
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
channel = list(range(7))
depth = np.arange(rdata.shape[1])*10.0 + 10.0

# calculate distributions
np.seterr(all="raise")
sp.special.seterr(all="raise")
gparam = np.zeros((numdata, len(channel), 4))
rparam = np.zeros((numdata, len(channel), 4))

gind = []
rind = []
for ii in range(numdata):
    for jj in range(len(channel)):
        try:
            gparam[ii,jj,:] = src.analysis.gaisser_hillas_fit(depth, gdata[ii,:,jj])
        except:
            gparam[ii,jj,:] = np.nan
            gind.append((ii,jj))
            print("gparam fit error at: ", (ii,jj))
        try:
            rparam[ii,jj,:] = src.analysis.gaisser_hillas_fit(depth, rdata[ii,:,jj])
        except:
            rparam[ii,jj,:] = np.nan
            rind.append((ii,jj))
            print("rparam fit error at: ", (ii,jj))

if len(gind) != 0 or len(rind) != 0:
    print("fit error at index:")
    print("gind = ", gind)
    print("rind = ", rind)


np.save(os.path.join(gaisser_hillas_path, "gdata_cond.npy"), gdata, fix_imports=False)
np.save(os.path.join(gaisser_hillas_path, "rdata_cond.npy"), rdata, fix_imports=False)
np.save(os.path.join(gaisser_hillas_path, "label_cond.npy"), label, fix_imports=False)
np.save(os.path.join(gaisser_hillas_path, "gfitparam.npy"), gparam, fix_imports=False)
np.save(os.path.join(gaisser_hillas_path, "rfitparam.npy"), rparam, fix_imports=False)
with open(os.path.join(gaisser_hillas_path, "fitparam_metadata.json"), "w") as fp:
    json.dump(
        {
            "fitparam_layout": {
                "nmax": 0,
                "xmax": 1,
                "x0": 2,
                "lam": 3,
                }
        },
        fp,
        indent = 4
    )

