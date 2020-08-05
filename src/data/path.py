import os
import src.utils

datapath = os.path.join(src.utils.get_root_path(), "data")

def get_path():
    """Return data path, i.e. repobase/data."""
    return datapath

def set_path(path):
    """Set data path. Default is repobase/data."""
    global datapath
    datapath = os.path.expanduser(path)

