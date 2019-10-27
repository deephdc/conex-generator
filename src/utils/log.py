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
        logging.getLogger().setLevel(level)

    if name is None:
        return logging.getLogger()

    return logging.getLogger(name)

