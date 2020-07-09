import numpy as np
import uuid
import os
import src.utils
from . import get_run_path

log = src.utils.getLogger(__name__)


def read_long_file(filepath):
    filename = os.path.split(filepath)[-1]
    log.info(f"reading long file {filename}")
    with open(filepath, "r") as file:
        lines = file.readlines()

    particle_distribution_list = []
    energy_deposit_list = []
    curline = 0
    while curline < len(lines):
        if "DISTRIBUTION" not in lines[curline]:
            error = "bad format in file " + filepath + \
                    " line " + str(curline)
            log.error(error)
            raise AssertionError(error)
        curline += 1
        if "DEPTH" not in lines[curline]:
            error = "bad format in file " + filepath + \
                    " line " + str(curline)
            log.error(error)
            raise AssertionError(error)
        curline += 1

        # read particle distribution
        particle_distribution = []
        while (curline < len(lines)) and ("SHOWER" not in lines[curline]):
            line = lines[curline]

            temp = [float(item) for item in filter(None, line.split(" "))]
            if len(temp) != 10:
                error = "bad format in file " + filepath + \
                        " line " + str(curline)
                log.error(error)
                raise AssertionError(error)

            particle_distribution.append(temp)
            curline += 1
        particle_distribution_list.append(particle_distribution)

        if "ENERGY" not in lines[curline]:
            error = "bad format in file " + filepath + \
                    " line " + str(curline)
            log.error(error)
            raise AssertionError(error)
        curline += 1
        if "DEPTH" not in lines[curline]:
            error = "bad format in file " + filepath + \
                    " line " + str(curline)
            log.error("error")
            raise AssertionError(error)
        curline += 1

        # read energy deposit
        energy_deposit = []
        while (curline < len(lines)) and ("SHOWER" not in lines[curline]):
            line = lines[curline]
            temp = [float(item) for item in filter(None, line.split(" "))]
            if len(temp) != 10:
                error = "bad format in file " + filepath + \
                        " line " + str(curline)
                log.error(error)
                raise AssertionError(error)
            energy_deposit.append(temp)
            curline += 1
        energy_deposit_list.append(energy_deposit)

    pd_list = np.transpose(particle_distribution_list, (0,2,1))
    ed_list = np.transpose(energy_deposit_list, (0,2,1))
    return (pd_list, ed_list)


def make_dataobject(particle, energy, theta, phi, obslevel,
                    particle_distribution_list, energy_deposit_list):
    if len(particle_distribution_list) != len(energy_deposit_list):
        msg = "particle distribution and energy deposit " \
              + "have different numbers of datapoints"
        log.error(msg)
        raise AssertionError(msg)

    # make dataobject
    log.info("creating data object")
    dataobject = {
            str(uuid.uuid4()): {
                "particle": particle,
                "energy": energy,
                "theta": theta,
                "phi": phi,
                "obslevel": obslevel,

                "particle_distribution": {
                    "depth":     particle_distribution_list[ii,0,0:-2].tolist(),
                    "gamma":     particle_distribution_list[ii,1,0:-2].tolist(),
                    "positron":  particle_distribution_list[ii,2,0:-2].tolist(),
                    "electron":  particle_distribution_list[ii,3,0:-2].tolist(),
                    "mup":       particle_distribution_list[ii,4,0:-2].tolist(),
                    "mum":       particle_distribution_list[ii,5,0:-2].tolist(),
                    "hadron":    particle_distribution_list[ii,6,0:-2].tolist(),
                    "charged":   particle_distribution_list[ii,7,0:-2].tolist(),
                    "nuclei":    particle_distribution_list[ii,8,0:-2].tolist(),
                    "cherenkov": particle_distribution_list[ii,9,0:-2].tolist(),
                },
                "energy_deposit": {
                    "depth":     energy_deposit_list[ii,0,0:-2].tolist(),
                    "gamma":     energy_deposit_list[ii,1,0:-2].tolist(),
                    "em_ioniz":  energy_deposit_list[ii,2,0:-2].tolist(),
                    "em_cut":    energy_deposit_list[ii,3,0:-2].tolist(),
                    "mu_ioniz":  energy_deposit_list[ii,4,0:-2].tolist(),
                    "mu_cut":    energy_deposit_list[ii,5,0:-2].tolist(),
                    "ha_ioniz":  energy_deposit_list[ii,6,0:-2].tolist(),
                    "ha_cut":    energy_deposit_list[ii,7,0:-2].tolist(),
                    "neutrino":  energy_deposit_list[ii,8,0:-2].tolist(),
                    "sum":       energy_deposit_list[ii,9,0:-2].tolist(),
                },
                
                "cutbin": {
                    "particle_distribution": {
                        "depth":     particle_distribution_list[ii,0,-2:].tolist(),
                        "gamma":     particle_distribution_list[ii,1,-2:].tolist(),
                        "positron":  particle_distribution_list[ii,2,-2:].tolist(),
                        "electron":  particle_distribution_list[ii,3,-2:].tolist(),
                        "mup":       particle_distribution_list[ii,4,-2:].tolist(),
                        "mum":       particle_distribution_list[ii,5,-2:].tolist(),
                        "hadron":    particle_distribution_list[ii,6,-2:].tolist(),
                        "charged":   particle_distribution_list[ii,7,-2:].tolist(),
                        "nuclei":    particle_distribution_list[ii,8,-2:].tolist(),
                        "cherenkov": particle_distribution_list[ii,9,-2:].tolist(),
                    },
                    "energy_deposit": {
                        "depth":     energy_deposit_list[ii,0,-2:].tolist(),
                        "gamma":     energy_deposit_list[ii,1,-2:].tolist(),
                        "em_ioniz":  energy_deposit_list[ii,2,-2:].tolist(),
                        "em_cut":    energy_deposit_list[ii,3,-2:].tolist(),
                        "mu_ioniz":  energy_deposit_list[ii,4,-2:].tolist(),
                        "mu_cut":    energy_deposit_list[ii,5,-2:].tolist(),
                        "ha_ioniz":  energy_deposit_list[ii,6,-2:].tolist(),
                        "ha_cut":    energy_deposit_list[ii,7,-2:].tolist(),
                        "neutrino":  energy_deposit_list[ii,8,-2:].tolist(),
                        "sum":       energy_deposit_list[ii,9,-2:].tolist(),
                    },
                },
            }
            for ii in range(len(particle_distribution_list))
    }

    return dataobject


def get_long_filepath(run):
    runpath = get_run_path()
    filename = "DAT{0:06d}.long".format(run)
    return os.path.join(runpath, filename)


def remove_long_file(run):
    filepath = get_long_filepath(run)
    filename = os.path.split(filepath)[-1]

    if os.path.isfile(filepath):
        log.info(f"removing steering file {filename}")
        os.remove(filepath)
    else:
        log.warning(f"cannot remove {filename} because it does not exist")

