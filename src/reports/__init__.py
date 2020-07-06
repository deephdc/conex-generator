from .path import get_path, set_path

import os
import lazy_import
if "lazy_import" in os.environ:
    lazy_import.lazy_module("src.reports.figures")

from . import figures

