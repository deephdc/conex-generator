import os
import src.utils

modelspath = os.path.join(src.utils.get_root_path(), "models")

def get_path():
    """Return models path, i.e. repobase/models.
    
    Returns
    -------
    datapath : str
    """
    return modelspath

def set_path(path):
    """Set models path. Default is repobase/models."""
    global modelspath
    modelspath = os.path.expanduser(path)

