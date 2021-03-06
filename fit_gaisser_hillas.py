"""Fit a 4 parameter Gaisser Hillas function to generated and real data.

Example: python3 fit_gaisser_hillas.py 1e8 1e10 35 70 \
         models/gan/run03/01
"""


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "ecut",
        nargs=2,
        type=float,
        help="lower and upper bound of the energy cut in GeV"
        )
parser.add_argument(
        "tcut",
        nargs=2,
        type=float,
        help="lower and upper bound of the theta cut in deg"
        )
parser.add_argument(
        "path",
        type=str,
        help="model data path relative to the project root directory"
        )
parser.add_argument(
        "--suffix",
        default="",
        type=str,
        help="data file suffix"
        )
parser.add_argument(
        "--fit-error",
        action="store_const",
        const=True,
        default=False,
        help="set nan on fit error"
        )

args = parser.parse_args()
ecut = args.ecut
tcut = args.tcut
model_base = args.path
file_suffix = args.suffix
fit_error = args.fit_error


import numpy as np
import scipy as sp
import os
import json

import src

root_path = src.utils.get_root_path()
model_path = os.path.join(root_path, model_base)
gaisser_hillas_path = os.path.join(
        model_path,
        "gaisser_hillas" \
                + file_suffix \
                + f"_ecut_{ecut[0]:.2e}_{ecut[1]:.2e}" \
                + f"_tcut_{tcut[0]:.2e}_{tcut[1]:.2e}" \
                + f"_fit-error_{fit_error}"
        )
if not os.path.isdir(gaisser_hillas_path):
    os.mkdir(gaisser_hillas_path)

gdata = np.load(os.path.join(model_path, "gdata" + file_suffix + ".npy"))
rdata = np.load(os.path.join(model_path, "rdata" + file_suffix + ".npy"))
label = np.load(os.path.join(model_path, "label" + file_suffix + ".npy"))

# prepare data
condition1 = label[:,0] >= 2
print("primary condition: ", np.sum(condition1))
condition2 = np.logical_and(label[:,1] >= ecut[0], label[:,1] < ecut[1])
print("energy condition: ", np.sum(condition2))
condition3 = np.logical_and(label[:,2] >= tcut[0], label[:,2] < tcut[1])
print("theta condition: ", np.sum(condition3))
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
if fit_error:
    np.seterr(all="raise")
    sp.special.seterr(all="raise")

gparam = np.zeros((numdata, len(channel), 4))
rparam = np.zeros((numdata, len(channel), 4))

for ii in range(numdata):
    for jj in range(len(channel)):
        try:
            gparam[ii,jj,:] = src.analysis.gaisser_hillas_fit(depth, gdata[ii,:,jj])
            rparam[ii,jj,:] = src.analysis.gaisser_hillas_fit(depth, rdata[ii,:,jj])
        except KeyboardInterrupt:
            exit(0)
        except:
            gparam[ii,jj,:] = np.nan
            rparam[ii,jj,:] = np.nan
            print("param fit error at: ", (ii,jj))

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
                },
            "energy_cut": ecut,
            "theta_cut": tcut,
            "fit_error": fit_error,
        },
        fp,
        indent = 4
    )

