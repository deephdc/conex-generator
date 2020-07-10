import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "nruns",
        type=int,
        help="number of corsika runs"
        )
parser.add_argument(
        "nshower",
        type=int,
        help="number of showers per corsika run"
        )
parser.add_argument(
        "nthreads",
        type=int,
        help="number of threads used"
        )
parser.add_argument(
        "store",
        type=str,
        help="raw folder at which data will be stored"
        )
parser.add_argument(
        "--binary",
        type=str,
        default="EPOS",
        help="corsika binary (glob-)expression"
        )
parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="corsika version"
        )
args = parser.parse_args()

numruns = args.nruns
showerperrun = args.nshower
numthreads = args.nthreads
rawfolder = args.store
binary = args.binary
version = args.version

import src
import random
random.seed()

log = src.utils.getLogger(__name__)
src.steering.corsika.set_version(version)
src.steering.corsika.set_binary(binary)
log.warning("running corsika from %s", src.steering.corsika.get_binary())

particle_list = src.steering.corsika.get_particle_list()
particle_list.pop("gamma", None)
particle_list.pop("electron", None)
print(particle_list)

src.steering.corsika.clean_env()

# physical parameters
energy_range = [17, 19]
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
src.data.raw.store_data(alldata, rawfolder)

log.warning("make_data.py done")

