import os
import json
import numpy as np
import gc
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_path, "config.json"), "r") as stream:
    config = json.load(stream)

data_path = config["data_path"]
json_merge_file = config["json_merge_file"]


# read json data
try:
    filename = sys.argv[1]
except:
    filename = os.path.join(data_path, json_merge_file)

with open(filename, "r") as fd:
    print("reading json data: ", filename)
    jsondata = json.load(fd)
gc.collect()

numdata = len(jsondata)


# get max depthlen
print("finding maximum depth length")
depthlen = 0
for value in jsondata.values():
    curdepthlen = len(value["particle_distribution"]["depth"])
    if curdepthlen > depthlen:
        depthlen = curdepthlen
gc.collect()


# write numpy data
print("converting to numpy data")
particle_list_in = config["particle_list_in"]
print("using incident particle list: ", particle_list_in)

numpy_data_layout = config["numpy_data_layout"]
print("using numpy data layout: ", numpy_data_layout)

numpy_label_layout = config["numpy_label_layout"]
print("using numpy label layout: ", numpy_label_layout)

data = np.zeros([numdata, depthlen, len(numpy_data_layout)])
label = np.zeros([numdata, len(numpy_label_layout)])

for ii, value in enumerate(jsondata.values()):
    particle_distribution = value["particle_distribution"]
    curdepthlen = len(particle_distribution["depth"])

    for curkey, curindex in numpy_data_layout.items():
        data[ii, 0:curdepthlen, curindex] = particle_distribution[curkey]

    for curkey, curindex in numpy_label_layout.items():
        if curkey == "particle":
            label[ii, curindex] = particle_list_in[value["particle"]]
        else:
            label[ii, curindex] = value[curkey]

    del(particle_distribution)


# write numpy data
print("writing numpy data to disk")
filename = os.path.join(data_path, "data.npy")
np.save(filename, data, fix_imports=False)
filename = os.path.join(data_path, "label.npy")
np.save(filename, label, fix_imports=False)

