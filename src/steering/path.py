import os
import src.utils

install_path = "/cr/users/koepke/install"

def get_install_path():
    """Return installation path where external programs reside.

    Returns
    -------
    datapath : str
    """
    return install_path

def set_install_path(path):
    """Set installation path where external programs reside.

    Parameters
    ----------
    path : str
    """
    global install_path
    install_path = os.path.expanduser(path)

