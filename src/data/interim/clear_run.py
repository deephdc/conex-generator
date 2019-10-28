import shutil
import os

import src.utils
log = src.utils.getLogger(__name__)

from . import get_path
curpath = get_path()


def clear_run(run):
    rmpath = os.path.join(curpath, run)
    if os.path.isdir(rmpath):
        log.info("removing %s", rmpath)
        shutil.rmtree(rmpath)

