import numpy as np
from scipy.spatial.distance import cdist
import ot


def wasserstein_distance_1d(histograms, centers, normalize=True, **kwargs):
    # handle args
    if not isinstance(centers[0], np.ndarray):
        centers = (centers, centers)

    metric = kwargs.pop("metric", "euclidean")

    # checks
    assert len(histograms) == 2
    assert histograms[0].ndim == 1
    assert histograms[1].ndim == 1

    assert len(centers) == 2
    assert centers[0].ndim == 1
    assert centers[1].ndim == 1

    # point distance
    centers0 = centers[0][...,np.newaxis]
    centers1 = centers[1][...,np.newaxis]
    M = cdist(centers0, centers1, metric=metric)

    # normalize histograms
    if normalize:
        hist0 = histograms[0] / histograms[0].sum()
        hist1 = histograms[1] / histograms[1].sum()
    else:
        hist0 = histograms[0]
        hist1 = histograms[1]

    # wasserstein distance
    distance = ot.emd2(hist0, hist1, M)
    return distance


def wasserstein_distance_2d(histograms, xcenters, ycenters, normalize=True, **kwargs):
    # handle args
    if not isinstance(xcenters[0], np.ndarray):
        xcenters = (xcenters, xcenters)
    if not isinstance(ycenters[0], np.ndarray):
        ycenters = (ycenters, ycenters)

    metric = kwargs.pop("metric", "euclidean")

    # checks
    assert len(histograms) == 2
    assert histograms[0].ndim == 2
    assert histograms[1].ndim == 2

    assert len(xcenters) == 2
    assert xcenters[0].ndim == 1
    assert xcenters[1].ndim == 1

    assert len(ycenters) == 2
    assert ycenters[0].ndim == 1
    assert ycenters[1].ndim == 1

    # point distance
    xx0, yy0 = np.meshgrid(xcenters[0], ycenters[0], indexing="ij")
    xx1, yy1 = np.meshgrid(xcenters[1], ycenters[1], indexing="ij")

    centers0 = np.stack([xx0.flatten(), yy0.flatten()], axis=-1)
    centers1 = np.stack([xx1.flatten(), yy1.flatten()], axis=-1)
    M = cdist(centers0, centers1, metric=metric)

    # normalize histograms
    if normalize:
        hist0 = (histograms[0] / histograms[0].sum()).flatten()
        hist1 = (histograms[1] / histograms[1].sum()).flatten()
    else:
        hist0 = histograms[0].flatten()
        hist1 = histograms[1].flatten()

    # wasserstein distance
    distance = ot.emd2(hist0, hist1, M)
    return distance

