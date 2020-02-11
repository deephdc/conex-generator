import logging
import uuid
import os

from . import get_root_path
root_path = get_root_path()
log_path = os.path.join(root_path, "log.txt")

from .telegram import TelegramHandler

logging.basicConfig(
        level = logging.WARNING,
        format = f"%(levelname)-1.1s/%(asctime)s/{uuid.uuid4()}/%(name)s: %(message)s",
        handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(),
            TelegramHandler()
            ]
        )

#disable tensorflow cpp logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def getLogger(name = None, level = None):
    if level is not None:
        setLogLevel(level)

    if name is None:
        return logging.getLogger()

    return logging.getLogger(name)


def setLogLevel(level : str):
    level = level.lower()
    logger = logging.getLogger()

    if level == "critical" or level == "c":
        logger.setLevel(logging.CRITICAL)
        return
    if level == "error" or level == "e":
        logger.setLevel(logging.ERROR)
        return
    if level == "warning" or level == "w":
        logger.setLevel(logging.WARNING)
        return
    if level == "info" or level == "i":
        logger.setLevel(logging.INFO)
        return
    if level == "debug" or level == "d":
        logger.setLevel(logging.DEBUG)
        return
    if level == "notset" or level == "n":
        logger.setLevel(logging.NOTSET)
        return

    raise RuntimeError("unsupported logging level")

