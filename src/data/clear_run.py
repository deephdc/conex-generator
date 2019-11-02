from .interim import clear_run as clear_interim
from .processed import clear_run as clear_processed


def clear_run(run):
    clear_interim(run)
    clear_processed(run)

