import numpy as np
import uuid
import src.utils

log = src.utils.getLogger(__name__)


def read_long_file(datfilepath):
    with open(datfilepath, "r") as file:
        lines = file.readlines()

    particle_distribution_list = []
    energy_deposit_list = []
    curline = 0
    while curline < len(lines):
        if "DISTRIBUTION" not in lines[curline]:
            error = "bad format in file " + datfilepath + \
                    " line " + str(curline)
            log.error(error)
            raise AssertionError(error)
        curline += 1
        if "DEPTH" not in lines[curline]:
            error = "bad format in file " + datfilepath + \
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
                error = "bad format in file " + datfilepath + \
                        " line " + str(curline)
                log.error(error)
                raise AssertionError(error)

            particle_distribution.append(temp)
            curline += 1
        particle_distribution_list.append(particle_distribution)

        if "ENERGY" not in lines[curline]:
            error = "bad format in file " + datfilepath + \
                    " line " + str(curline)
            log.error(error)
            raise AssertionError(error)
        curline += 1
        if "DEPTH" not in lines[curline]:
            error = "bad format in file " + datfilepath + \
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
                error = "bad format in file " + datfilepath + \
                        " line " + str(curline)
                log.error(error)
                raise AssertionError(error)
            energy_deposit.append(temp)
            curline += 1
        energy_deposit_list.append(energy_deposit)

    return (particle_distribution_list, energy_deposit_list)


def make_dataobject(particle, energy, theta, phi, obslevel,
                    particle_distribution_list, energy_deposit_list):
    if len(particle_distribution_list) != len(energy_deposit_list):
        msg = "particle distribution and energy deposit " \
              + "have different numbers of datapoints"
        log.error(msg)
        raise AssertionError(msg)

    pd_list = np.transpose(particle_distribution_list, (0,2,1))
    ed_list = np.transpose(energy_deposit_list, (0,2,1))

    # make dataobject
    dataobject = {
            str(uuid.uuid4()): {
                "particle": particle,
                "energy": energy,
                "theta": theta,
                "phi": phi,
                "obslevel": obslevel,
                "particle_distribution": {
                    "depth":     pd_list[ii,0].tolist(),
                    "gamma":     pd_list[ii,1].tolist(),
                    "positron":  pd_list[ii,2].tolist(),
                    "electron":  pd_list[ii,3].tolist(),
                    "mup":       pd_list[ii,4].tolist(),
                    "mum":       pd_list[ii,5].tolist(),
                    "hadron":    pd_list[ii,6].tolist(),
                    "charged":   pd_list[ii,7].tolist(),
                    "nuclei":    pd_list[ii,8].tolist(),
                    "cherenkov": pd_list[ii,9].tolist(),
                },
                "energy_deposit": {
                    "depth":     ed_list[ii,0].tolist(),
                    "gamma":     ed_list[ii,1].tolist(),
                    "em_ioniz":  ed_list[ii,2].tolist(),
                    "em_cut":    ed_list[ii,3].tolist(),
                    "mu_ioniz":  ed_list[ii,4].tolist(),
                    "mu_cut":    ed_list[ii,5].tolist(),
                    "ha_ioniz":  ed_list[ii,6].tolist(),
                    "ha_cut":    ed_list[ii,7].tolist(),
                    "neutrino":  ed_list[ii,8].tolist(),
                    "sum":       ed_list[ii,9].tolist(),
                },
            }
            for ii in range(len(pd_list))
    }

    return dataobject

