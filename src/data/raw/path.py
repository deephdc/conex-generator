import os

from .. import get_path as get_parent_path

def get_path():
    return os.path.join(get_parent_path(), "raw")

