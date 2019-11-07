import os
import json

script_path = os.path.dirname(os.path.realpath(__file__))
numpy_config_path = os.path.join(script_path, "raw/json_to_numpy/config.json")

with open(numpy_config_path, "r") as fp:
    numpy_config_dict = json.load(fp)


def numpy_config():
    return dict(numpy_config_dict)

def numpy_particle_label():
    return dict(numpy_config_dict["particle_label"])

def numpy_data_layout():
    return dict(numpy_config_dict["numpy_data_layout"])

def numpy_label_layout():
    return dict(numpy_config_dict["numpy_label_layout"])

