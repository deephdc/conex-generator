import matplotlib
import matplotlib.pyplot as plt
import logging

import src
log = src.utils.getLogger(__name__, "info")

run = "run01"
src.data.clear_run(run + "*")
src.data.raw.convert_run(run, expand_depth=True)
src.data.interim.align_run(run, expand_depth=True)
src.data.interim.merge_run(run, expand_depth=True)

