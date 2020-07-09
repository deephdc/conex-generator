import os
import glob

from .. import get_install_path


search_expr = os.path.join(get_install_path(), "corsika-*")
corsika_paths = glob.glob(search_expr)
corsika_paths.sort()
corsika_path = corsika_paths[-1]

corsika_filename = "corsika77000Linux_QGSII_urqmd_thin_conex"


def get_path():
    return corsika_path

def set_version(version):
    global corsika_path
    newpath = os.path.join(get_install_path(), "corsika-" + version)
    if os.path.isdir(newpath):
        corsika_path = newpath
    else:
        raise RuntimeError(f"cannot find version corsika-{version}")

def get_run_path():
    return os.path.join(get_path(), "run")

def get_corsika_binary():
    runpath = get_run_path()
    
    return os.path.join(runpath, corsika_filename)

def set_corsika_binary(binary):
    global corsika_filename
    binary = os.path.split(binary)[-1]

    if os.path.isfile(os.path.join(get_run_path(), binary)):
        corsika_filename = binary
    else:
        raise RuntimeError(f"cannot find corsika binary {binary}")

