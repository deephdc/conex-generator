import os

from .. import get_path as get_parent_path

def get_path():
    """Return data interim path, i.e. repobase/data/interim.
    
    Returns
    -------
    datapath : str
    """
    return os.path.join(get_parent_path(), "interim")

