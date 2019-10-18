import numpy as np
import uuid

def readlongfile(datfilename):
    with open(datfilename, "r") as file:
        lines = file.readlines()

    particle_distribution_list = []
    energy_deposit_list = []
    curline = 0
    while curline < len(lines):
        if "DISTRIBUTION" not in lines[curline]:
            error = "bad format in file " + datfilename + \
                    " line " + str(curline)
            raise AssertionError(error)
        curline += 1
        if "DEPTH" not in lines[curline]:
            error = "bad format in file " + datfilename + \
                    " line " + str(curline)
            raise AssertionError(error)
        curline += 1

        # read particle distribution
        particle_distribution = []
        while (curline < len(lines)) and ("SHOWER" not in lines[curline]):
            line = lines[curline]
            temp = [float(item) for item in filter(None, line.split(" "))]
            assert len(temp) == 10
            particle_distribution.append(temp)
            curline += 1
        particle_distribution_list.append(particle_distribution)

        if "ENERGY" not in lines[curline]:
            error = "bad format in file " + datfilename + \
                    " line " + str(curline)
            raise AssertionError(error)
        curline += 1
        if "DEPTH" not in lines[curline]:
            error = "bad format in file " + datfilename + \
                    " line " + str(curline)
            raise AssertionError(error)
        curline += 1

        # read energy deposit
        energy_deposit = []
        while (curline < len(lines)) and ("SHOWER" not in lines[curline]):
            line = lines[curline]
            temp = [float(item) for item in filter(None, line.split(" "))]
            assert len(temp) == 10
            energy_deposit.append(temp)
            curline += 1
        energy_deposit_list.append(energy_deposit)

    return (particle_distribution_list, energy_deposit_list)


def makedataobject(particle, energy, theta, phi,
                   particle_distribution_list, energy_deposit_list):
    assert len(particle_distribution_list) == len(energy_deposit_list)

    particle_distribution_list = np.transpose(particle_distribution_list, (0,2,1))
    energy_deposit_list = np.transpose(energy_deposit_list, (0,2,1))

    # make dataobject
    dataobject = {
            str(uuid.uuid4()): {
                "particle": particle,
                "energy": energy,
                "theta": theta,
                "phi": phi,
                "particle_distribution": {
                    "depth":     particle_distribution_list[ii,0].tolist(),
                    "gamma":     particle_distribution_list[ii,1].tolist(),
                    "positron":  particle_distribution_list[ii,2].tolist(),
                    "electron":  particle_distribution_list[ii,3].tolist(),
                    "mup":       particle_distribution_list[ii,4].tolist(),
                    "mum":       particle_distribution_list[ii,5].tolist(),
                    "hadron":    particle_distribution_list[ii,6].tolist(),
                    "charged":   particle_distribution_list[ii,7].tolist(),
                    "nuclei":    particle_distribution_list[ii,8].tolist(),
                    "cherenkov": particle_distribution_list[ii,9].tolist(),
                },
                "energy_deposit": {
                    "depth":     energy_deposit_list[ii,0].tolist(),
                    "gamma":     energy_deposit_list[ii,1].tolist(),
                    "em_ioniz":  energy_deposit_list[ii,2].tolist(),
                    "em_cut":    energy_deposit_list[ii,3].tolist(),
                    "mu_ioniz":  energy_deposit_list[ii,4].tolist(),
                    "mu_cut":    energy_deposit_list[ii,5].tolist(),
                    "ha_ioniz":  energy_deposit_list[ii,6].tolist(),
                    "ha_cut":    energy_deposit_list[ii,7].tolist(),
                    "neutrino":  energy_deposit_list[ii,8].tolist(),
                    "sum":       energy_deposit_list[ii,9].tolist(),
                },
            }
            for ii in range(len(particle_distribution_list))
    }

    return dataobject

