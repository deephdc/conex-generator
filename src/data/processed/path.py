import os

from .. import get_path as get_parent_path

def get_path():
    """Return data processed path, i.e. repobase/data/raw.
    
    Returns
    -------
    datapath : str
    """
    return os.path.join(get_parent_path(), "processed")

