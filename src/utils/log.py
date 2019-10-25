import logging

logging.basicConfig(
        level = logging.WARNING,
        format = "%(levelname)-1.1s/%(asctime)s/%(name)s: %(message)s",
        handlers = [
            logging.FileHandler("log.txt"),
            logging.StreamHandler(),
            ]
        )

def getLogger(name = None, level = None):
    if level is not None:
        logging.getLogger().setLevel(level)

    if name is None:
        return logging.getLogger()

    return logging.getLogger(name)

