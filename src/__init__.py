from . import utils
from .utils.path import get_src_path as get_path

import os
import lazy_import
if "lazy_import" in os.environ:
    lazy_import.lazy_module("src.data")
    lazy_import.lazy_module("src.models")
    lazy_import.lazy_module("src.reports")
    lazy_import.lazy_module("src.analysis")
    lazy_import.lazy_module("src.steering")
    lazy_import.lazy_module("src.plot")

from . import data
from . import models
from . import reports
from . import analysis
from . import steering
from . import plot

