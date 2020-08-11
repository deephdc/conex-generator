"""Package that handles steering of external programs such as CORSIKA."""

from .path import get_install_path, set_install_path

import os
import lazy_import
if "lazy_import" in os.environ:
    lazy_import.lazy_module("src.steering.corsika")

from . import corsika

