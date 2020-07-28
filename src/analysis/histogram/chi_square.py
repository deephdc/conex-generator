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

