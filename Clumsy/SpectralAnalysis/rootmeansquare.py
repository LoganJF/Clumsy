import numpy as np
from numpy import (sqrt, mean, square, floor, zeros)

__all__ = [
    'root_mean_square',
    'RMS_amplitude_detector',
    'RMS_amplitude_detector_fast',
    'valid_rms_index'
]
def rolling_window(arr, window):
    """Returns an array of windowed sections of passed array a

    Parameters
    ----------
    a: array, array to get windows of
    window: int, length of window

    Returns
    -------
    windowed array

    See Also http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def root_mean_square(arr, axis=-1):
    """Calculates a root mean square on inputted array

    Parameters
    ----------
    arr: np.array, array to get root mean square of
    axis: int, by default -1, axis to apply root mean square along

    Returns
    -------
    root mean square of the array
    """
    # from numpy import (mean, sqrt, square)
    return sqrt(mean(square(arr), axis))


def RMS_amplitude_detector_fast(arr, duration=.2, fs=100.):
    """Applies a root mean square sliding window along an inputted array

    Parameters
    ----------
    arr: np.array, array to get root mean square of
    duration: float, desired duration in seconds
    fs: float, sample rate of array

    Returns
    -------
    root mean square of array
    """
    # from numpy import floor
    win = int(floor(fs * duration))
    rms = root_mean_square(rolling_window(np.array(arr), window=win))
    return rms


def valid_rms_index(rms, percentile=95):
    indx_above_thres = np.where(rms >= np.percentile(rms, percentile))
    return indx_above_thres


# SO SLOW!!!!
def RMS_amplitude_detector(data, duration=.2, fs=100.):
    """Computes a moving root mean square"""
    # from numpy import (floor, mean, sqrt, square, zeros)
    halfdur = int(floor(fs * duration / 2))
    lendat = len(data)
    rms = zeros((lendat))
    arr = data.data  # numpy array faster than timeseries indexing
    for i in range(lendat):
        rms[i] = sqrt(mean(square(
            arr[max(0, i - halfdur):min(lendat, i + halfdur)])))
    return rms