import os
import sys
import json
import glob
import gc

script_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_path, "config.json"), "r") as stream:
    config = json.load(stream)

globexpr = os.path.join(config["data_path"], "*/*.json")
filenames = glob.glob(globexpr)
print("all files:")
for filename in filenames:
    print("\t ", filename)
print()

data = {}
for filename in filenames:
    print("current file: ", filename)
    with open(filename, "r") as file:
        temp = json.load(file)
        data.update(temp)
    del(temp)
    gc.collect()
print()

new_filename = os.path.join(config["data_path"], config["json_merge_file"])
with open(new_filename, "w") as file:
    print("writing data file")
    json.dump(data, file)

