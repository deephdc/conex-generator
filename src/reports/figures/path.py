import os

from .. import get_path as get_parent_path

def get_path():
    """Return figures path, i.e. repobase/reports/figures.
    
    Returns
    -------
    path : str
    """
    return os.path.join(get_parent_path(), "figures")

