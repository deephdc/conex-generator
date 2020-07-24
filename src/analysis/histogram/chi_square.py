import numpy as np


def chi_square_distance(histograms, normalize=True, **kwargs):
    #checks
    assert len(histograms) == 2
    assert histograms[0].shape == histograms[1].shape

    # normalize histograms
    if normalize:
        hist0 = histograms[0] / histograms[0].sum()
        hist1 = histograms[1] / histograms[1].sum()
    else:
        hist0 = histograms[0]
        hist1 = histograms[1]

    hdiff = hist0 - hist1
    hsum  = hist0 + hist1
    hsum = np.where(hsum == 0, 1, hsum)
    hquotient = np.square(hdiff) / hsum

    return np.sum(hquotient) / 2


if __name__ == "__main__":
    import generators

    xdata1 = np.random.uniform(0,1,10000).reshape(10000,1)
    ydata1 = np.random.uniform(0,1,10000).reshape(10000,1)
    xdata2 = np.random.uniform(0,1,10000).reshape(10000,1)
    ydata2 = np.random.uniform(0,1,10000).reshape(10000,1)

    retval = generators.correlation_histogram_generator(xdata1, ydata1, xdata2, ydata2)

    for ret in retval:
        chi2 = chi_square_distance(**ret)
        print(chi2)

