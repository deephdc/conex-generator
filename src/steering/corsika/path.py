import os
import glob

from .. import get_install_path


def get_path():
    search_expr = os.path.join(get_install_path(), "corsika-*")
    corsika_paths = glob.glob(search_expr)
    corsika_paths.sort()

    return corsika_paths[-1]


def get_run_path():
    return os.path.join(get_path(), "run")


def get_corsika_path():
    runpath = get_run_path()
    filename = "corsika77000Linux_QGSII_gheisha_thin_conex"
    
    return os.path.join(runpath, filename)

