import os
import src.utils

datapath = os.path.join(src.utils.get_root_path(), "data")

def get_path():
    return datapath

def set_path(path):
    global datapath
    datapath = os.path.expanduser(path)

