import lazy_import

from .path import get_install_path, set_install_path

lazy_import.lazy_module("src.steering.corsika")
from . import corsika

