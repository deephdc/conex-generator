import shutil
import os
import glob

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
curpath = get_path()


def clear_run(run):
    """Remove all derived data files from data/processed.

    Parameters
    ----------
    run : str
        Name of the subfolder in data/processed.
        Accepts glob expressions.
    """
    rmpath_glob = os.path.join(curpath, run)
    rmpaths = glob.glob(rmpath_glob)
    for path in rmpaths:
        if os.path.isdir(path):
            log.info("removing %s", path)
            shutil.rmtree(path)

