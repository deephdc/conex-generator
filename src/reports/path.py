import os
import src.utils

reportspath = os.path.join(src.utils.get_root_path(), "reports")

def get_path():
    """Return reports path, i.e. repobase/reports.
    
    Returns
    -------
    path : str
    """
    return reportspath

def set_path(path):
    """Set reports path. Default is repobase/reports."""
    global reportspath
    reportspath = os.path.expanduser(path)

