import subprocess
import os
import glob
import src.utils

from . import get_run_path, get_corsika_path, get_long_filepath
from . import write_steering_file
from . import read_long_file
from .long_file import make_dataobject
from .steering_file import remove_steering_file
from .long_file import remove_long_file

log = src.utils.getLogger(__name__)


def call(
        particle,
        energy,
        theta,
        phi,
        obslevel = 0.0,
        nshower = 1,
        run = None):

    runpath = get_run_path()
    coriska_path = get_corsika_path()
    coriska_file = os.path.split(coriska_path)[-1]
    run = find_run(run)

    long_filepath = get_long_filepath(run)
    long_filename = os.path.split(long_filepath)[-1]
    if os.path.isfile(long_filepath):
        msg = f"cannot call coriska because long file {long_filename} exists"
        log.error(msg)
        raise FileExistsError(msg)

    steering_filepath = write_steering_file(
            particle=particle, energy=energy, theta=theta, phi=phi,
            obslevel=obslevel, nshower=nshower, run=run, overwrite=False)

    log.info(f"running {coriska_file} at {runpath} with run number {run}")
    shellcmd = os.path.join(".", coriska_file)
    with open(steering_filepath, "r") as infile:
        retval = subprocess.call(
                [shellcmd],
                stdin=infile,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=runpath)
    if retval != 0:
        msg = f"corsika exit code {retval}"
        log.error(msg)
        raise RuntimeError(msg)

    pd_list, ed_list = read_long_file(long_filepath)
    remove_long_file(run)
    remove_steering_file(run)

    return (pd_list, ed_list)


def get_data(
        particle,
        energy,
        theta,
        phi,
        obslevel = 0.0,
        nshower = 1,
        run = None):

    pd_list, ed_list = call(particle,
                            energy,
                            theta,
                            phi,
                            obslevel,
                            nshower,
                            run)

    dataobject = make_dataobject(particle,
                                 energy,
                                 theta,
                                 phi,
                                 obslevel,
                                 pd_list,
                                 ed_list)

    return dataobject


def find_run(run):
    if run is not None:
        return run
    
    runpath = get_run_path()
    pattern = os.path.join(runpath, "*_conex.cfg")
    files = glob.glob(pattern)
    cfgnumbers = [
            int(os.path.split(curfile)[-1].split("_conex.cfg")[0])
            for curfile in files
            ]
    cfgnumbers.append(-1)

    pattern = os.path.join(runpath, "DAT*.long")
    files = glob.glob(pattern)
    longnumbers = [
            int(os.path.split(curfile)[-1].split(".long")[0].split("DAT")[-1])
            for curfile in files
            ]
    longnumbers.append(-1)

    currun = max([max(cfgnumbers), max(longnumbers)]) + 1
    log.info(f"using automatic run number {currun}")
    return currun

