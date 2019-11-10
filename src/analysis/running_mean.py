import numpy as np


def running_mean(x, N):
    if N <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

