import os

utils_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.split(utils_path)[0]
root_path = os.path.split(src_path)[0]


def get_root_path():
    return root_path


def get_src_path():
    return src_path


def get_utils_path():
    return utils_path

