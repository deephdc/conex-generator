import numpy as np

from .generators import correlation_histogram_generator
from .wasserstein_distance import wasserstein_distance_2d
from .chi_square import chi_square_distance


supported_distances = {
        "wasserstein": wasserstein_distance_2d,
        "chi2": chi_square_distance,
}


def compare_correlation(
        xdata1, ydata1,
        xdata2, ydata2,
        metrics=["wasserstein"],
        bins=30,
        sparse=False):
    # checks
    assert xdata1.shape == xdata2.shape
    assert ydata1.shape == ydata2.shape
    assert xdata1.ndim == 2
    assert ydata1.ndim == 2

    if sparse:
        assert xdata1.shape == ydata1.shape

    # create histograms
    histgen = correlation_histogram_generator(
            xdata1, ydata1,
            xdata2, ydata2,
            bins=bins,
            sparse=sparse)

    # matrix to store all metric values
    nummetrics = len(metrics)
    xnumchannel = xdata1.shape[1]
    ynumchannel = ydata1.shape[1]

    mmatrix = np.full(
            (xnumchannel, ynumchannel, nummetrics),
            np.nan,
            dtype=np.float)

    # loop over histograms and metrics
    for retval in histgen:
        for mindex, metric in enumerate(metrics):
            xindex = retval["xindex"]
            yindex = retval["yindex"]

            distance_func = supported_distances[metric]
            distance = distance_func(**retval)

            mmatrix[xindex, yindex, mindex] = distance

    if sparse:
        mmatrixT = np.transpose(mmatrix, axes=(1,0,2))
        mmatrix = np.where(np.isnan(mmatrix), mmatrixT, mmatrix)

    return {
            metric: mmatrix[..., mindex]
            for mindex, metric in enumerate(metrics)
    }

