import matplotlib
import matplotlib.pyplot as plt
import logging

import src
log = src.utils.getLogger(__name__, logging.INFO)

#test = src.data.raw.convert_run("run02", expand_depth=False)

src.data.interim.merge_run("run02", expand_depth=False, overwrite=True)

