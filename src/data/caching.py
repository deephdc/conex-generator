import os
import glob
import timeit

import src.utils
log = src.utils.getLogger(__name__)


def prepare_cache(name, basepath="/home/tmp/koepke/cache", reuse=True):
    if not os.path.exists(basepath):
        log.info("creating cache folder at: %s", basepath)
        os.makedirs(basepath)

    path = os.path.join(basepath, name)

    files = glob.glob(path + "*")
    if len(files) != 0 and not reuse:
        log.info("removing old cache: %s", name)
        for f in files:
            os.remove(f)

    return path


def cache_dataset(dataset, name, basepath="/home/tmp/koepke/cache",
                  reuse=True, cachenow=True):
    path = prepare_cache(name, basepath=basepath, reuse=reuse)
    files = glob.glob(path + "*")

    ds = dataset.cache(path)
    if len(files) == 0 and cachenow:
        log.info("writing cache to disc: %s", path)
        start = timeit.default_timer()
        for x in iter(ds):
            pass
        end = timeit.default_timer()
        log.info("cache time: %.2f seconds", end-start)

    return ds

