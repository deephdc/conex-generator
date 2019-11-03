import os
import json

script_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_path, "config.json"), "r") as fp:
    config = json.load(fp)


def get_config():
    return dict(config)


def get_particle_list():
    return dict(config["particle_list"])

