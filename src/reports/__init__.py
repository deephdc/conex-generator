import lazy_import

from .path import get_path, set_path

lazy_import.lazy_module("src.reports.figures")
from . import figures

