import os
import json

script_path = os.path.dirname(os.path.realpath(__file__))
numpy_config_path = os.path.join(script_path, "raw/json_to_numpy/config.json")

with open(numpy_config_path, "r") as fp:
    numpy_config_dict = json.load(fp)


def numpy_config():
    """Return full numpy data config dictionary.

    The config is stored in "src/data/raw/json_to_numpy/config.json".
    """
    return dict(numpy_config_dict)

def numpy_particle_label():
    """Return numpy data label dictionary.

    The dictionary contains the assignment of readable names and numeric
    values for the particle lables.
    The config is stored in "src/data/raw/json_to_numpy/config.json".
    """
    return dict(numpy_config_dict["particle_label"])

def numpy_data_layout():
    """Return numpy data layout dictionary.

    The dictionary contains the assignment of readable names and numeric
    indices for the particle distribution and energy deposit data files.
    The config is stored in "src/data/raw/json_to_numpy/config.json".
    """
    return dict(numpy_config_dict["numpy_data_layout"])

def numpy_label_layout():
    """Return numpy data layout dictionary.

    The dictionary contains the assignment of readable names and numeric
    indices for the label data files.
    The config is stored in "src/data/raw/json_to_numpy/config.json".
    """
    return dict(numpy_config_dict["numpy_label_layout"])

