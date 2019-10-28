import matplotlib
import matplotlib.pyplot as plt
import logging

import src
log = src.utils.getLogger(__name__, logging.INFO)

run = "run02"
#src.data.interim.clear_run(run)
#src.data.processed.clear_run(run)
#src.data.raw.convert_run(run, expand_depth=False)
#src.data.interim.merge_run(run, expand_depth=False)

dataset, metadata = src.data.processed.load_data(run)

