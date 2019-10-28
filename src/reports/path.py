import os
import src.utils

reportspath = os.path.join(src.utils.get_root_path(), "reports")

def get_path():
    return reportspath

def set_path(path):
    global reportspath
    reportspath = os.path.expanduser(path)

