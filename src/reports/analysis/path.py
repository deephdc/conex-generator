import os

from .. import get_path as get_parent_path

def get_path():
    """Return analysis path, i.e. repobase/reports/analysis.
    
    Returns
    -------
    path : str
    """
    return os.path.join(get_parent_path(), "analysis")

