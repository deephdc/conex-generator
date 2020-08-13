"""General package that handles common utility functions like logging etc."""

from .path import get_utils_path as get_path
from .path import get_root_path
from .timestamp import timestamp, timestamp_utc_iso, timestamp_local_iso
from .log import getLogger, setLogLevel
from .byte_string_to_number import parse_byte_string

