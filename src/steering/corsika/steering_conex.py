import random
random.seed()

import threading
import queue
import traceback
import json
from datetime import datetime

import os
import sys

from changevalues import writefile
from readlongfile import readlongfile, makedataobject

dir_path = os.path.dirname(os.path.realpath(__file__))
corsika_path = os.path.join(os.environ["CORSIKA_DIR"], "run")
os.chdir(corsika_path)

numruns = int(sys.argv[1])
numthreads = int(sys.argv[2])
try:
    showerperrun = int(sys.argv[3])
except:
    showerperrun = 10
try:
    startrun = int(sys.arg[4])
except:
    startrun = 0

# physical parameters
energy_range = [1e8, 1e10]
theta_range = [0.0, 65.0]
phi_range = [-180.0, 180.0]
particle_list = {
        1:    "gamma",
        3:    "electron",
        14:   "proton",
        402:  "helium",
        1608: "oxygen",
        5626: "iron",
}


def worker_call(threadid, jobq : queue.Queue, dataq : queue.Queue, seedint):
    random.seed(seedint)
    while True:
        run = jobq.get()

        energy = random.uniform(*energy_range)
        theta = random.uniform(*theta_range)
        phi = random.uniform(*phi_range)
        particle = random.choice(list(particle_list.keys()))

        writefile(run=run, number=showerperrun, particle=particle,
                  energy=energy, theta=theta, phi=phi)

        datfilename = "DAT{0:06d}.long".format(run)
        os.system("rm -f " + datfilename)
        os.system("./corsika77000Linux_QGSII_gheisha_thin_conex < " + \
                  str(run) + "_conex.cfg > /dev/null")

        try:
            particle_distribution_list , energy_deposit_list = readlongfile(datfilename)
            dataobject = makedataobject(particle_list[particle], energy, theta,
                                        phi, particle_distribution_list,
                                        energy_deposit_list)
            dataq.put(dataobject)
        except AssertionError:
            traceback.print_exc()
            jobq.put(run)
        except:
            pass

        jobq.task_done()

# create queues
jobqueue = queue.Queue()
[jobqueue.put(inp) for inp in range(startrun, startrun+numruns)]
dataqueue = queue.Queue()

# start threads
for curthreadid in range(numthreads):
    seedint = random.randrange(0, int(2**32))
    worker = threading.Thread(target=worker_call,
                              args=(curthreadid, jobqueue, dataqueue, seedint,))
    worker.setDaemon(True)
    worker.start()

# wait for finish
try:
    jobqueue.join()
except KeyboardInterrupt:
    pass

# join all data
alldata = {}
while not dataqueue.empty():
    curdata = dataqueue.get()
    alldata.update(curdata)

datafilename = "conex_data_"+str(datetime.now())+".json"
datafilename = datafilename.replace(" ", "-").replace(":", "-")
with open(datafilename, "w") as file:
    json.dump(alldata, file)

