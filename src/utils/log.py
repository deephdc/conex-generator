import logging
import uuid

logging.basicConfig(
        level = logging.WARNING,
        format = f"%(levelname)-1.1s/%(asctime)s/{uuid.uuid4()}/%(name)s: %(message)s",
        handlers = [
            logging.FileHandler("log.txt"),
            logging.StreamHandler(),
            ]
        )

def getLogger(name = None, level = None):
    if level is not None:
        setLevel(level)

    if name is None:
        return logging.getLogger()

    return logging.getLogger(name)


def setLevel(level : str):
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

