import logging

logging.basicConfig(
        level = logging.DEBUG,
        format = "%(levelname)-1.1s/%(asctime)s/%(name)s: %(message)s",
        handlers = [
            logging.FileHandler("log.txt"),
            logging.StreamHandler(),
            ]
        )

def getLogger(*args, **kwargs):
    return logging.getLogger(*args, **kwargs)

