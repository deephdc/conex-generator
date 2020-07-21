import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot
import matplotlib.pyplot as plt


def histogram_generator(data1, data2, bins=30):
    # checks
    assert data1.shape == data2.shape
    assert data1.ndim == 2

    if bins < 4:
        raise ValueError("cannot operate with less than 4 bins")

    # calc bins
    lower1 = np.min(data1, axis=0)
    lower2 = np.min(data2, axis=0)
    lower  = np.min([lower1, lower2], axis=0)

    upper1 = np.max(data1, axis=0)
    upper2 = np.max(data2, axis=0)
    upper  = np.max([upper1, upper2], axis=0)

    hrange = upper - lower
    bwidth = hrange / (bins - 2)

    bin_edges = np.linspace(lower-bwidth, upper+bwidth, bins+1)

    # yield histograms
    for ii in range(data1.shape[1]):
        hist1, edges1 = np.histogram(data1[:,ii], bins=bin_edges[:,ii], density=True)
        hist2, edges2 = np.histogram(data2[:,ii], bins=bin_edges[:,ii], density=True)
        centers = (bin_edges[1:,ii] + bin_edges[:-1,ii])/2

        yield {
                "histograms": (hist1.astype(np.float), hist2.astype(np.float)),
                "centers": centers.astype(np.float),
                "edges": bin_edges[:,ii].astype(np.float),
                "index": ii,
        }


def correlation_histogram_generator(xdata1, ydata1, xdata2, ydata2, bins=30, sparse=False):
    # checks
    assert xdata1.shape == xdata2.shape
    assert ydata1.shape == ydata2.shape
    assert xdata1.ndim == 2
    assert ydata1.ndim == 2

    if sparse:
        assert xdata1.shape == ydata1.shape

    if bins < 4:
        raise ValueError("cannot operate with less than 4 bins")

    # calc x bins
    xlower1 = np.min(xdata1, axis=0)
    xlower2 = np.min(xdata2, axis=0)
    xlower  = np.min([xlower1, xlower2], axis=0)

    xupper1 = np.max(xdata1, axis=0)
    xupper2 = np.max(xdata2, axis=0)
    xupper  = np.max([xupper1, xupper2], axis=0)

    xhrange = xupper - xlower
    xbwidth = xhrange / (bins - 2)

    xbin_edges = np.linspace(xlower-xbwidth, xupper+xbwidth, bins+1)

    # calc y bins
    ylower1 = np.min(ydata1, axis=0)
    ylower2 = np.min(ydata2, axis=0)
    ylower  = np.min([ylower1, ylower2], axis=0)

    yupper1 = np.max(ydata1, axis=0)
    yupper2 = np.max(ydata2, axis=0)
    yupper  = np.max([yupper1, yupper2], axis=0)

    yhrange = yupper - ylower
    ybwidth = yhrange / (bins - 2)

    ybin_edges = np.linspace(ylower-ybwidth, yupper+ybwidth, bins+1)

    # yield histograms
    for ii in range(xdata1.shape[1]):
        for jj in range(ydata1.shape[1]):
            if sparse and jj > ii:
                break

            hist1, xedges1, yedges1 = np.histogram2d(
                    xdata1[:,ii],
                    ydata1[:,jj],
                    bins=[xbin_edges[:,ii], ybin_edges[:,jj]],
                    density=True)

            hist2, xedges2, yedges2 = np.histogram2d(
                    xdata2[:,ii],
                    ydata2[:,jj],
                    bins=[xbin_edges[:,ii], ybin_edges[:,jj]],
                    density=True)

            xcenters = (xbin_edges[1:,ii] + xbin_edges[:-1,ii])/2
            ycenters = (ybin_edges[1:,jj] + ybin_edges[:-1,jj])/2

            yield {
                    "histograms": (hist1.astype(np.float), hist2.astype(np.float)),
                    "xcenters": xcenters.astype(np.float),
                    "ycenters": ycenters.astype(np.float),
                    "xedges": xbin_edges[:,ii].astype(np.float),
                    "yedges": ybin_edges[:,jj].astype(np.float),
                    "index": (ii, jj),
            }


if __name__ == "__main__":
    shape = (10000, 1)
    npoints = np.product(shape)

    xdata1 = np.arange(1000).reshape(1000,1)
    ydata1 = np.arange(1000).reshape(1000,1)

    xdata2 = np.arange(1000).reshape(1000,1)
    ydata2 = np.arange(1000).reshape(1000,1)

    retval = correlation_histogram_generator(xdata1, ydata1, xdata2, ydata2)
    retval = list(retval)

    hists = retval[0]["histograms"]
    xcenters = retval[0]["xcenters"]
    ycenters = retval[0]["ycenters"]

    xedges = retval[0]["xedges"]
    yedges = retval[0]["yedges"]

    plt.figure()
    plt.pcolormesh(xedges, yedges, hists[0].T)
    plt.figure()
    plt.pcolormesh(xedges, yedges, hists[1].T)

    #M = cdist(centers.reshape(30,1), centers.reshape(30,1))
    #wd = ot.emd2(hists[0], hists[1], M)
    #print(wd)

