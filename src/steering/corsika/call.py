import subprocess
import os
import glob
import threading
import queue
import traceback
import src.utils

from . import get_run_path, get_binary
from . import get_long_filepath, get_steering_filepath
from . import write_steering_file
from . import read_long_file
from .config import get_particle_list
from .long_file import make_dataobject
from .steering_file import remove_steering_file
from .long_file import remove_long_file

log = src.utils.getLogger(__name__)

particle_list = get_particle_list()


def call(
        particle,
        energy,
        theta,
        phi,
        obslevel = 0.0,
        nshower = 1,
        run = None,
        clean = False):
    """Run CORSIKA with a given set of inputs and return longitudinal data.

    This function calls CORSIKA with a given default steering file called
    "conex.default" (in src/steering/coriska). It replaces the parts about
    particle type, energy, zenith, azimuth and observation level before it
    does so.

    Parameters
    ----------
    particle : str
        Particle type as string representation. Currently supports
            "gamma" (CORSIKA 1)
            "electron" (CORSIKA 3)
            "proton" (CORSIKA 14)
            "helium" (CORSIKA 402)
            "oxygen" (CORSIKA 1608)
            "iron" (CORSIKA 5626)
    energy : float
        Incident energy in GeV.
    theta : float
        Zenith angle in degree. Range: [0, 90].
    phi : float
        Azimuth angle in degree. Range: [-180, 180].
    obslevel : float, optional
        Observation level for CORSIKA in cm. Should usually be greater than
        zero, however some atmospheres allow -10^5 cm as a minimum.
        Defaults to 0.0.
    nshower : int, optinal
        Batch size for the CORSIKA run, i.e. how many profiles will be
        generated. Defaults to 1.
    run : int, optional
        Internal CORSIKA run number that fixes the numeric suffix of output
        files. Defaults to None, which will use the next free number.
    clean : bool, optional
        Flag to indicate if the output file and steering file should be
        removed after a successful execution. Defaults to False.

    Returns
    -------
    (pd_list, ed_list) : tuple
        Contents of the longitudinal file stored as a numpy array. pd_list
        is the particle_distribution and ed_list is the energy_deposit. Both
        layouts are (batch, channel, depth).
    """
    runpath = get_run_path()
    coriska_path = get_binary()
    coriska_file = os.path.split(coriska_path)[-1]
    run = find_run(run)

    long_filepath = get_long_filepath(run)
    long_filename = os.path.split(long_filepath)[-1]
    if os.path.isfile(long_filepath):
        msg = f"cannot call coriska because long file {long_filename} exists"
        log.error(msg)
        raise FileExistsError(msg)

    write_steering_file(
            particle=particle_list[particle], energy=energy, theta=theta,
            phi=phi, obslevel=obslevel, nshower=nshower, run=run,
            overwrite=False)

    log.info(f"running {coriska_file} at {runpath} with run number {run}")
    shellcmd = os.path.join(".", coriska_file)
    steering_filepath = get_steering_filepath(run)
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

    if clean:
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
        run = None,
        clean = False):
    """Run CORSIKA with a given set of inputs and return a dataobject.

    This function calls CORSIKA with a given default steering file called
    "conex.default" (in src/steering/coriska). It replaces the parts about
    particle type, energy, zenith, azimuth and observation level before it
    does so. The return type also contains metadata information (besides the
    longitudinal profile).

    Parameters
    ----------
    particle : str
        Particle type as string representation. Currently supports
            "gamma" (CORSIKA 1)
            "electron" (CORSIKA 3)
            "proton" (CORSIKA 14)
            "helium" (CORSIKA 402)
            "oxygen" (CORSIKA 1608)
            "iron" (CORSIKA 5626)
    energy : float
        Incident energy in GeV.
    theta : float
        Zenith angle in degree. Range: [0, 90].
    phi : float
        Azimuth angle in degree. Range: [-180, 180].
    obslevel : float, optional
        Observation level for CORSIKA in cm. Should usually be greater than
        zero, however some atmospheres allow -10^5 cm as a minimum.
        Defaults to 0.0.
    nshower : int, optional
        Batch size for the CORSIKA run, i.e. how many profiles will be
        generated. Defaults to 1.
    run : int, optional
        Internal CORSIKA run number that fixes the numeric suffix of output
        files. Defaults to None, which will use the next free number.
    clean : bool, optional
        Flag to indicate if the output file and steering file should be
        removed after a successful execution. Defaults to False.

    Returns
    -------
    dataobject : dict
        A dataobject that contains the longitudinal particle distrubtion,
        energy deposit and metadata. The last two depth bins are stored in
        "cutbin" because CORSIKA treats them in a special way and they are
        not part of an ordinary profile shape.
        
        The exact format is (see also src.steering.corsika.make_dataobject):

        {
            str(uuid.uuid4()): {
                "particle": particle,
                "energy": energy,
                "theta": theta,
                "phi": phi,
                "obslevel": obslevel,

                "particle_distribution": {
                    "depth":
                    "gamma":
                    "positron":
                    "electron":
                    "mup":
                    "mum":
                    "hadron":
                    "charged":
                    "nuclei":
                    "cherenkov":
                },
                "energy_deposit": {
                    "depth":
                    "gamma":
                    "em_ioniz":
                    "em_cut":
                    "mu_ioniz":
                    "mu_cut":
                    "ha_ioniz":
                    "ha_cut":
                    "neutrino":
                    "sum":
                },
                
                "cutbin": {
                    "particle_distribution": {
                        "depth":
                        "gamma":
                        "positron":
                        "electron":
                        "mup":
                        "mum":
                        "hadron":
                        "charged":
                        "nuclei":
                        "cherenkov":
                    },
                    "energy_deposit": {
                        "depth":
                        "gamma":
                        "em_ioniz":
                        "em_cut":
                        "mu_ioniz":
                        "mu_cut":
                        "ha_ioniz":
                        "ha_cut":
                        "neutrino":
                        "sum":
                    },
                },
            },

            # ... for each shower in a batch
        }
    """
    pd_list, ed_list = call(particle,
                            energy,
                            theta,
                            phi,
                            obslevel,
                            nshower,
                            run,
                            clean)

    dataobject = make_dataobject(particle,
                                 energy,
                                 theta,
                                 phi,
                                 obslevel,
                                 pd_list,
                                 ed_list)

    return dataobject


def get_data_distributed(
        particle,
        energy,
        theta,
        phi,
        obslevel,
        nshower,
        startrun = None,
        clean = False,
        nthreads = 4):
    """Run many CORSIKA instances in parallel and return a dataobject.

    This function calls CORSIKA in parallel with a given default steering file
    called "conex.default" (in src/steering/coriska). It replaces the parts
    about particle type, energy, zenith, azimuth and observation level before
    it does so. The number of calls to CORSIKA is determined by the length of
    the input lists. The return type also contains metadata information
    (besides the longitudinal profile).

    Parameters
    ----------
    particle : list(str)
        List of particle types as string representation. Currently supports
            "gamma" (CORSIKA 1)
            "electron" (CORSIKA 3)
            "proton" (CORSIKA 14)
            "helium" (CORSIKA 402)
            "oxygen" (CORSIKA 1608)
            "iron" (CORSIKA 5626)
    energy : list(float)
        List of incident energies in GeV.
    theta : list(float)
        List of zenith angles in degree. Range: [0, 90].
    phi : list(float)
        List of azimuth angles in degree. Range: [-180, 180].
    obslevel : list(float)
        List of observation levels for CORSIKA in cm. Should usually be greater
        than zero, however some atmospheres allow -10^5 cm as a minimum.
    nshower : list(int)
        list of batch sizes for the CORSIKA runs, i.e. how many profiles will
        be generated in each run.
    startrun : int, optional
        Internal CORSIKA run number that fixes the numeric suffix of output
        files. Every single run will have an incremented run number starting
        with this parameter.
        Defaults to None, which will use the next free number.
    clean : bool, optional
        Flag to indicate if the output files and steering files should be
        removed after the full successful execution of all runs.
        Defaults to False.
    nthreads : int, optional
        How many instances of CORSIKA should be run in parallel to work
        through the lists of inputs. Defaults to 4.

    Returns
    -------
    dataobject : dict
        A dataobject that contains the longitudinal particle distrubtion,
        energy deposit and metadata. The last two depth bins are stored in
        "cutbin" because CORSIKA treats them in a special way and they are
        not part of an ordinary profile shape.
        
        The exact format is (see also src.steering.corsika.make_dataobject):

        {
            str(uuid.uuid4()): {
                "particle": particle,
                "energy": energy,
                "theta": theta,
                "phi": phi,
                "obslevel": obslevel,

                "particle_distribution": {
                    "depth":
                    "gamma":
                    "positron":
                    "electron":
                    "mup":
                    "mum":
                    "hadron":
                    "charged":
                    "nuclei":
                    "cherenkov":
                },
                "energy_deposit": {
                    "depth":
                    "gamma":
                    "em_ioniz":
                    "em_cut":
                    "mu_ioniz":
                    "mu_cut":
                    "ha_ioniz":
                    "ha_cut":
                    "neutrino":
                    "sum":
                },
                
                "cutbin": {
                    "particle_distribution": {
                        "depth":
                        "gamma":
                        "positron":
                        "electron":
                        "mup":
                        "mum":
                        "hadron":
                        "charged":
                        "nuclei":
                        "cherenkov":
                    },
                    "energy_deposit": {
                        "depth":
                        "gamma":
                        "em_ioniz":
                        "em_cut":
                        "mu_ioniz":
                        "mu_cut":
                        "ha_ioniz":
                        "ha_cut":
                        "neutrino":
                        "sum":
                    },
                },
            },

            # ... for each shower in a batch and for each run
        }
    """
    # prepare and check args
    len_particle = len(particle)
    len_energy = len(energy)
    len_theta = len(theta)
    len_phi = len(phi)
    len_obslevel = len(obslevel)
    len_nshower = len(nshower)
    len_all = [
            len_particle,
            len_energy,
            len_theta,
            len_phi,
            len_obslevel,
            len_nshower,
            ]
    if max(len_all) != min(len_all):
        msg = "inputs must be of same length: particle, energy, theta, phi," \
              + " obslevel, nshower"
        log.error(msg)
        raise AssertionError(msg)

    ndata = max(len_all)
    run = find_run(startrun)

    # create temporary configs to reserve run numbers
    runpath = get_run_path()
    log.info(f"writing temporary config for run number {run} to {run+ndata-1}")
    for ii in range(ndata):
        currun = run + ii
        filepath = os.path.join(runpath, str(currun) + "_conex.cfg")
        if os.path.isfile(filepath):
            msg = f"cannot distribute corika because run number {currun}" \
                  + " does already exist"
            log.error(msg)
            raise FileExistsError(msg)
        with open(filepath, "w") as fp:
            fp.write("pending")

    # fill jobqueue
    jobqueue = queue.Queue()
    dataqueue = queue.Queue()
    for ii in range(ndata):
        currun = run + ii
        kwargs = {
                "particle": particle[ii],
                "energy": energy[ii],
                "theta": theta[ii],
                "phi": phi[ii],
                "obslevel": obslevel[ii],
                "nshower": nshower[ii],
                "run": currun,
                "clean": False
                }
        jobqueue.put(kwargs, timeout=10)

    
    for ii in range(nthreads):
        worker = threading.Thread(target=distributed_worker,
                                  args=(jobqueue,dataqueue))
        worker.setDaemon(True)
        worker.start()

    # prefetch data
    alldata = {}
    while jobqueue.qsize() != 0:
        try:
            dataobject = dataqueue.get(timeout=1)
            alldata.update(dataobject)
        except queue.Empty:
            pass

    try:
        jobqueue.join()
    except KeyboardInterrupt:
        pass

    while dataqueue.qsize() != 0:
        try:
            dataobject = dataqueue.get(timeout=1)
            alldata.update(dataobject)
        except queue.Empty:
            pass

    if clean:
        for ii in range(ndata):
            currun = run + ii
            remove_long_file(currun)
            remove_steering_file(currun)

    if len(alldata) != sum(nshower):
        log.warning("length of returned data does not match requested length")

    return alldata


def find_run(run):
    """Find the next free run number in install/coriska-*/run.

    This function checks the run directory of the CORSIKA binary for a free
    (internal) run number such that the output file suffix or steering file
    prefix is not in use yet.

    Parameters
    ----------
    run : int
        Minimal start number for the run number search. Can be None, which
        will indicate the use of the next available number.

    Returns
    -------
    currun : int
        An available run number.
    """
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


def distributed_worker(jobqueue : queue.Queue, dataqueue : queue.Queue):
    """The worker function for src.steering.corsika.get_data_distributed.

    This worker function implements a retry automatic, which will requeue a
    failed CORSIKA call once.

    Paramters
    ---------
    jobqueue : queue.Queue
        A queue which holds the input information for CORSIKA calls.
    dataqueue : queue.Queue
        A queue which holds the ouput dataobjects.
    """
    while True:
        try:
            kwargs = dict(jobqueue.get(timeout=10))
        except queue.Empty:
            log.debug("no more jobs. stopping thread")
            return

        currun = kwargs["run"]
        retry = kwargs.pop("retry", False)
        
        try:
            # remove temporary config
            filepath = get_steering_filepath(currun)
            if os.path.isfile(filepath):
                os.remove(filepath)

            # run corsika/conex
            dataobject = get_data(**kwargs)
            dataqueue.put(dataobject, timeout=10)
            log.info(f"run number {currun} successfully finished")
        except KeyboardInterrupt:
            log.debug("thread accepted KeyboardInterrupt")
            return
        except:
            traceback.print_exc()

            # remove long file
            filepath = get_long_filepath(currun)
            if os.path.isfile(filepath):
                os.remove(filepath)

            # start retry if not already
            if not retry:
                log.warning(f"run number {currun} failed. retry queued")
                kwargs["retry"] = True
                jobqueue.put(kwargs, timeout=10)
            else:
                log.warning(f"retry of run number {currun} failed")
        finally:
            jobqueue.task_done()

