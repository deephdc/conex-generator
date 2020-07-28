import numpy as np

from .generators import histogram_generator
from .wasserstein_distance import wasserstein_distance_1d
from .chi_square import chi_square_distance


supported_distances = {
        "wasserstein": wasserstein_distance_1d,
        "chi2": chi_square_distance,
}


def compare_histogram_1d(
        data1,
        data2,
        bins=30,
        metrics=["wasserstein"]):
    #checks
    assert data1.shape == data2.shape
    assert data1.ndim == 2

    # create histograms
    histgen = histogram_generator(
            data1,
            data2,
            bins=bins)

    # matrix to store all metric values
    nummetrics = len(metrics)
    numchannel = data1.shape[1]

    mmatrix = np.full(
            (numchannel, nummetrics),
            np.nan,
            dtype=np.float)

    # loop over histograms and metrics
    for retval in histgen:
        for mindex, metric in enumerate(metrics):
            index = retval["index"]

            distance_func = supported_distances[metric]
            distance = distance_func(**retval)

            mmatrix[index, mindex] = distance

    return {
            metric: mmatrix[..., mindex]
            for mindex, metric in enumerate(metrics)
    }

