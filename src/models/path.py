import os
import src.utils

modelspath = os.path.join(src.utils.get_root_path(), "models")

def get_path():
    return modelspath

def set_path(path):
    global modelspath
    modelspath = os.path.expanduser(path)

