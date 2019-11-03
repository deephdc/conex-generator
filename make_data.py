import sys

numruns = int(sys.argv[1])
showerperrun = int(sys.argv[2])
numthreads = int(sys.argv[3])

import src
import random
random.seed()

particle_list = src.steering.corsika.get_particle_list()
particle_list.pop("gamma", None)
particle_list.pop("electron", None)
print(particle_list)

src.steering.corsika.clean_env()

# physical parameters
energy_range = [17, 20]
theta_range = [0.0, 65.0]
phi_range = [-180.0, 180.0]
obslevel_range = [-1e5, -1e5]

# sample input
particle = [
        random.choice(list(particle_list.keys()))
        for ii in range(numruns)
]
energy = [
        (10**random.uniform(*energy_range)) / 1e9
        for ii in range(numruns)
]
theta = [
        random.uniform(*theta_range)
        for ii in range(numruns)
]
phi = [
        random.uniform(*phi_range)
        for ii in range(numruns)
]
obslevel = [
        random.uniform(*obslevel_range)
        for ii in range(numruns)
]
nshower = [
        showerperrun
        for ii in range(numruns)
]

# run conex
alldata = src.steering.corsika.get_data_distributed(
        particle,
        energy,
        theta,
        phi,
        obslevel,
        nshower,
        clean = True,
        nthreads = numthreads)

# store data
src.data.raw.store_data(alldata, "run03")

