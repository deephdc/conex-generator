import matplotlib
import matplotlib.pyplot as plt
import logging

import src
log = src.utils.getLogger(__name__, logging.INFO)

log.debug("debug message")
log.info("info message")
log.warning("warning message")
log.error("error message")
log.critical("critical message")

test = src.data.raw.json_to_numpy.create_dataobject("conex_data_2019-10-24-18-30-19.604496.json", "run02", expand_depth=False)
src.data.raw.json_to_numpy.store_dataobject(test, overwrite=True)

