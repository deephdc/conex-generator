import os
import random
from . import get_run_path
from . import get_steering_options
from . import get_binary
import src.utils

log = src.utils.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "conex.default"), "r") as file:
    orig_lines = list(file)


def write_steering_file(
        run = 1,
        nshower = 1,
        particle = 14,
        energy = 1e0,
        theta = 0.0,
        phi = 0.0,
        obslevel = 0.0,
        overwrite=False):
    runpath = get_run_path()
    filepath = get_steering_filepath(run)
    filename = os.path.split(filepath)[-1]
    if not overwrite and os.path.isfile(filepath):
        msg = "steering file " + filename + " does already exist"
        log.error(msg)
        raise FileExistsError(msg)

    new_lines = list(orig_lines)
    for ii in range(len(new_lines)):
        line = new_lines[ii]

        if "RUNNR" in line:
            new_lines[ii] ="{:7} {:<50}\n".format("RUNNR", run)
            continue

        if "NSHOW" in line:
            new_lines[ii] ="{:7} {:<50}\n".format("NSHOW", nshower)
            continue

        if "PRMPAR" in line:
            new_lines[ii] ="{:7} {:<50}\n".format("PRMPAR", particle)
            continue

        if "ERANGE" in line:
            new_lines[ii] = "{:7} {:<50}\n".format("ERANGE", "%.9e %.9e" % (energy, energy))
            continue

        if "THETAP" in line:
            new_lines[ii] = "{:7} {:<50}\n".format("THETAP", "%f %f" % (theta, theta))
            continue

        if "PHIP" in line:
            new_lines[ii] = "{:7} {:<50}\n".format("PHIP", "%f %f" % (phi, phi))
            continue

        if "SEED" in line:
            seed = [str(random.randrange(1, 900000000)), str(random.randrange(0, int(2**16))), "0"]
            seedstr = " ".join(seed)
            new_lines[ii] = "{:7} {:<50}\n".format("SEED", seedstr)
            continue
        
        if "OBSLEV" in line:
            new_lines[ii] = "{:7} {:<50}\n".format("OBSLEV", "%.9e" % (obslevel))
            continue

    # apply custom steering options
    steering_options = get_steering_options()
    for k, v in steering_options.items():
        if k in get_binary():
            new_lines = v + new_lines

    log.info(f"writing steering file {filename}")
    with open(filepath, "w") as file:
        file.writelines(new_lines)


def remove_steering_file(run):
    runpath = get_run_path()
    filepath = get_steering_filepath(run)
    filename = os.path.split(filepath)[-1]

    if os.path.isfile(filepath):
        log.info(f"removing steering file {filename}")
        os.remove(filepath)
    else:
        log.warning(f"cannot remove {filename} because it does not exist")


def get_steering_filepath(run):
    runpath = get_run_path()
    filepath = os.path.join(runpath, str(run) + "_conex.cfg")
    return filepath

