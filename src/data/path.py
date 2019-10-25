import os

datapath = os.path.expanduser("~/data/network/conex-generator/data")

def get_path():
    return datapath

def set_path(path):
    global _datapath
    datapath = os.path.expanduser(path)

