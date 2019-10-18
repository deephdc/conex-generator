import os
import random

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "conex.default"), "r") as file:
    orig_lines = list(file)

def writefile(path="./", run = 1, number = 1, particle = 14, energy = 1e0, theta = 0.0, phi = 0.0):
    new_lines = list(orig_lines)
    seedcounter = 1

    for ii in range(len(new_lines)):
        line = new_lines[ii]

        if "RUNNR" in line:
            new_lines[ii] ="{:7} {:<50}\n".format("RUNNR", run)
            continue

        if "NSHOW" in line:
            new_lines[ii] ="{:7} {:<50}\n".format("NSHOW", number)
            continue


        if "PRMPAR" in line:
            new_lines[ii] ="{:7} {:<50}\n".format("PRMPAR", particle)
            continue

        if "ERANGE" in line:
            new_lines[ii] = "{:7} {:<50}\n".format("ERANGE", "%.3e %.3e" % (energy, energy))
            continue

        if "THETAP" in line:
            new_lines[ii] = "{:7} {:<50}\n".format("THETAP", "%f %f" % (theta, theta))
            continue

        if "PHIP" in line:
            new_lines[ii] = "{:7} {:<50}\n".format("PHIP", "%f %f" % (theta, theta))
            continue

        if "SEED" in line:
            seed = [str(random.randrange(1, 900000000)), str(random.randrange(0, int(2**16))), "0"]
            seedstr = " ".join(seed)
            new_lines[ii] = "{:7} {:<50}\n".format("SEED", seedstr)
            continue
    
    filename = os.path.join(path, str(run) + "_conex.cfg")
    with open(filename, "w") as file:
        file.writelines(new_lines)

