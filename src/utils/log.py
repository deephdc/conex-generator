import logging
import uuid
import os

from . import get_root_path
root_path = get_root_path()
log_path = os.path.join(root_path, "log.txt")

from .telegram import TelegramHandler

logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
        level = logging.WARNING,
        format = f"%(levelname)-1.1s/%(asctime)s/{uuid.uuid4()}/%(name)s: %(message)s",
        handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(),
            TelegramHandler()
            ]
        )

#tensorflow cpp warning log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def getLogger(name = None, level = None):
    """Get a python (logging) logger.

    Uses logging.getLogger internally. The returned logger is attached to a
    file stream in repobase/log.txt, stdout and a Telegram-Bot (if possible).
    The default format is defined in src/utils/log.py.

    Paramters
    ---------
    name : str, optional
        Logger name. Should be called with __name__. Defaults to None, which
        will return the root logger.
    level : str, optional
        Sets the log level of the root logger (and therefore also for the
        derived logger). Should be one of the following:
            "critical" or "c"
            "error" or "e"
            "warning" or "w"
            "info" or "i"
            "debug" or "d"
            "notset" or "n"
            (case insensitive)
        Defaults to None, which will not change the log level.

    Returns
    -------
    logger : logging.Logger
    """
    if level is not None:
        setLogLevel(level)

    if name is None:
        return logging.getLogger()

    return logging.getLogger(name)


def setLogLevel(level : str):
    """Set the log level of the root logger (and thus for all derived loggers).

    This function also manipulates the log level of the TensorFlow c++ logger
    via the environment variable "TF_CPP_MIN_LOG_LEVEL".

    Parameters
    ----------
    level : str
        Sets the log level of the root logger (and therefore also for all
        derived loggers). Should be one of the following:
            "critical" or "c"
            "error" or "e"
            "warning" or "w"
            "info" or "i"
            "debug" or "d"
            "notset" or "n"
            (case insensitive)
    """
    level = level.lower()
    logger = logging.getLogger()

    if level == "critical" or level == "c":
        logger.setLevel(logging.CRITICAL)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        return
    if level == "error" or level == "e":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        logger.setLevel(logging.ERROR)
        return
    if level == "warning" or level == "w":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        logger.setLevel(logging.WARNING)
        return
    if level == "info" or level == "i":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        logger.setLevel(logging.INFO)
        return
    if level == "debug" or level == "d":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        logger.setLevel(logging.DEBUG)
        return
    if level == "notset" or level == "n":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        logger.setLevel(logging.NOTSET)
        return

    raise RuntimeError("unsupported logging level")

