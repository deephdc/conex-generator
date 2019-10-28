import matplotlib
import matplotlib.pyplot as plt
import logging

import src
log = src.utils.getLogger(__name__, logging.INFO)

#src.data.interim.clear_run("run02")
#src.data.processed.clear_run("run02")
#src.data.raw.convert_run("run02", expand_depth=False)
#src.data.interim.merge_run("run02", expand_depth=False)

dataset, metadata = src.data.processed.load_data("run02")

