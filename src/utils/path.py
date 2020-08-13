import os

utils_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.split(utils_path)[0]
root_path = os.path.split(src_path)[0]


def get_root_path():
    """Return the root path of the project (i.e. below src).

    Returns
    -------
    path : str
    """
    return root_path


def get_src_path():
    """Return the src path of the project (i.e. repobase/src).

    Returns
    -------
    path : str
    """
    return src_path


def get_utils_path():
    """Return the utils path (i.e. repobase/src/utils).

    Returns
    -------
    path : str
    """
    return utils_path

