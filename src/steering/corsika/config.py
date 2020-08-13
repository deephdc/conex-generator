import os
import json

script_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(script_path, "config.json"), "r") as fp:
    config = json.load(fp)


def get_config():
    """Return CORSIKA steering config.

    Returns
    -------
    config : dict
        The JSON dictionary stored in src/steering/corsika/config.json
    """
    return dict(config)


def get_particle_list():
    """Return CORSIKA particle name and number assignment.

    This is part of the config in src/steering/corsika/config.json.

    Returns
    -------
    particle_list : dict
        Particle dictionary. keys -> name, values -> number.
    """
    return dict(config["particle_list"])


def get_steering_options():
    """Return special steering options for specific CORSIKA modes.

    This is part of the config in src/steering/corsika/config.json.

    Returns
    -------
    options : dict
        Options dictionary. keys -> mode.
    """
    return dict(config["steering"])

