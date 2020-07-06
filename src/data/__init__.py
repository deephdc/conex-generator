from .path import get_path, set_path
from .clear_run import clear_run
from .config import numpy_config
from .config import numpy_particle_label
from .config import numpy_data_layout
from .config import numpy_label_layout
from .caching import prepare_cache
from .caching import cache_dataset

import os
import lazy_import
if "lazy_import" in os.environ:
    lazy_import.lazy_module("src.data.raw")
    lazy_import.lazy_module("src.data.interim")
    lazy_import.lazy_module("src.data.processed")
    lazy_import.lazy_module("src.data.external")
    lazy_import.lazy_module("src.data.random")
else:
    lazy_import.lazy_module("src.data.processed")
    lazy_import.lazy_module("src.data.random")

from . import raw
from . import interim
from . import processed
from . import external
from . import random

