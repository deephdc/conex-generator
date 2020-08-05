import os
import glob
import timeit

import src.utils
log = src.utils.getLogger(__name__)


def prepare_cache(name, basepath="/home/tmp/koepke/cache", reuse=True):
    """Delete cache path if it exists and reuse is False. Else create it.
    
    Because TensorFlow will automatically reuse a cached dataset and ignore
    the original one, it is necessary to check wheter or not to reuse it.
    This function just accomplishes that.
    
    Parameters
    ----------
    name : str
        Name of the dataset cache, e.g. "particle_distribution".
    basepath : str, optional
        Directory path in which the cache files will be created. Defaults to
        "/home/tmp/koepke/cache" because this is IKP default for SSD storage.
    reuse: bool, optional
        Flag to indicate if an already present cache should be reused. If not
        the old cache will be deleted. Defaults to True (do reuse cache).
    """
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
    """Cache a tf.data.Dataset instance on filesystem.

    This function adds immediate caching behavior to the tf.data.Dataset.cache
    functionality and enables recreation of an already present cache file.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset that should be cached.
    name : str
        Name of the dataset cache, e.g. "particle_distribution".
    basepath : str, optional
        Directory path in which the cache files will be created. Defaults to
        "/home/tmp/koepke/cache" because this is IKP default for SSD storage.
    reuse: bool, optional
        Flag to indicate if an already present cache should be reused. If not
        the old cache will be deleted. Defaults to True (do reuse cache).
    cachenow : bool, optional
        Due to TensorFlow internals a dataset cache will only be created when
        the dataset is iterated upon. This flag ensures that the whole dataset
        cache will be created immediatly. Defaults to True. Without any good
        reason this should never be changed.
    """
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

