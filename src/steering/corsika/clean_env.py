import os
import glob
import src.utils

from . import get_run_path

log = src.utils.getLogger(__name__)


def clean_env():
    runpath = get_run_path()
    log.info(f"cleaning corsika run environment at {runpath}")

    pattern = os.path.join(runpath, "DAT*.long")
    files = glob.glob(pattern)
    for curfile in files:
        os.remove(curfile)

    pattern = os.path.join(runpath, "*_conex.cfg")
    files = glob.glob(pattern)
    for curfile in files:
        os.remove(curfile)

