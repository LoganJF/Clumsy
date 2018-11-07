"""staging.py

The purpose of this script is allow one to automatically classify intracranial eeg data as either wake, Non-REM,
or not Non-REM. The way we calculate the stage is through the use of a metric called Visualization Index (VI).

Steps:
Window data into 30s epochs with 0s of overlap
Run a FFT over each 30s epoch
Normalize the resulting PSD across time and channels to 1 using the total area under the power spectrum.
Across subjects: normalize the scale so maximal VI value, for each channel, is equal to 100 and minimum is 1
Average all channels

Definitions:
                        delta = (1,4)
                        theta = (4, 8)
                        alpha = (8, 12)
                        spindle = (11,16)
                        high_beta = (20, 40)
                        VI = (delta + theta + spindle) / (high_beta, alpha)
"""

from Clumsy.timeseriesLF import TimeSeriesLF
import numpy as np
from Clumsy.signal import psd, rolling_window_full


__all__ = [
    'min_max_normalization',
    'integrate',
    'norm_by_integral',
    'normalize_to_high',
    'timeline',
    'valid_band',
    'get_VI'
]


def min_max_normalization(x, minimum, maximum):
    # Normalize channels to between .0 and +1.0
    y = (x - minimum) / (maximum - minimum)
    return y

def _get_VI(ts, window_len=30, overlap=0):
    """

    Parameters
    ----------
    ts: TimeSeries, timeseries data to analyze
    window_len: int, time in seconds to moved over
    overlap: int, time in seconds to overlap fft segments over

    Returns
    -------
    ts_fft_rolled
    """
    """Gets vigilance index defined as (Delt + Theta + Spindles)/ (alpha + high_beta) normalized between 0-100"""
    # window_len = 30
    step = ts._TimeSeries__duration_to_samples(window_len)
    asteps = ts._TimeSeries__duration_to_samples(window_len - overlap)
    arr = ts.data

    rolled = rolling_window_full(array=arr, window=step, asteps=asteps)
    # FFT data
    freq, data = psd(rolled, Fs=float(ts['samplerate'].data))

    coords = {k: v for k, v in ts.coords.items() if k != 'time'}
    coords['time'] = np.arange(0, rolled.shape[1] * window_len, window_len)
    coords['frequency'] = freq
    dims = ['channels', 'time', 'frequency']
    ts_fft_rolled = TimeSeriesLF(data, coords=coords, dims=dims)
    return ts_fft_rolled


def integrate(y, dx=1., axis=-1):

    """Calculates the discrete integral along the given axis using the composite trapezoidal rule

    Parameters
    ----------
    y: array-like
        input array for integral
    dx: float, default=1.,
        step over which to integrate on y
    axis: int, by default -1,
        axis over which to integrate

    Returns
    -------
    integral: array-like
              shape = (y.shape[:-1])
    """
    integral = np.trapz(y=y, dx=dx, axis=axis)
    return integral


def norm_by_integral(timeseries, dx=1., axis=-1):
    """Apply an integral to normalize a given timeseries

    Parameters
    ----------
    timeseries
    dx: float, default=1.,
        step over which to integrate on y
    axis: int, by default -1,
        axis over which to integrate

    Returns
    -------
    normalized
    """
    integral = integrate(timeseries, dx=dx, axis=axis)
    # Norm so each ch sums to 1
    normalized = timeseries / integral[:, None]
    return normalized


def timeline(start, step, amount):
    """Create a range of len(howmany) starting at start with a step of step

    Parameters
    ----------
    start:int,
        starting value
    step: int,
        step value
    amount: int,
        how long to make the array

    Returns
    -------

    """
    return np.arange(start, amount * step, step)


def valid_band(freq, low, high):
    """Given an array of increasing, frequencies return a boolean of locations

    Parameters
    ----------
    freq:
    low:
    high:

    Returns
    -------

    """
    return (freq >= low) & (freq <= high)


def get_VI(ts, window_len=30, overlap=0):
    """Gets vigilance index

     normalized between 0-100

     VI =  (Delt + Theta + Spindles)/ (alpha + high_beta)

    Parameters
    ----------
    ts
    window_len
    overlap

    Returns
    -------

    """
    """Gets vigilance index defined as (Delt + Theta + Spindles)/ (alpha + high_beta) normalized between 0-100"""
    # window_len = 30
    step = ts._TimeSeries__duration_to_samples(window_len)
    asteps = ts._TimeSeries__duration_to_samples(window_len - overlap)
    arr = ts.data

    rolled = rolling_window_full(array=arr, window=step, asteps=asteps)
    # FFT data
    freq, data = psd(rolled, Fs=float(ts['samplerate'].data))

    coords = {k: v for k, v in ts.coords.items() if k != 'time'}
    coords['time'] = np.arange(0, rolled.shape[1] * window_len, window_len)
    coords['frequency'] = freq
    dims = ['channels', 'time', 'frequency']
    ts_fft_rolled = TimeSeriesLF(data, coords=coords, dims=dims)
    # return ts_fft_rolled
    freq_arr = ts_fft_rolled.frequency.data
    delt = ts_fft_rolled.sel(frequency=ts_fft_rolled.frequency[valid_band(freq_arr, 1, 4)]).mean('frequency')
    theta = ts_fft_rolled.sel(frequency=ts_fft_rolled.frequency[valid_band(freq_arr, 4, 8)]).mean('frequency')
    alpha = ts_fft_rolled.sel(frequency=ts_fft_rolled.frequency[valid_band(freq_arr, 8, 12)]).mean('frequency')
    spindle = ts_fft_rolled.sel(frequency=ts_fft_rolled.frequency[valid_band(freq_arr, 11, 16)]).mean('frequency')
    high_beta = ts_fft_rolled.sel(frequency=ts_fft_rolled.frequency[valid_band(freq_arr, 20, 40)]).mean('frequency')
    VI = (delt + theta + spindle) / (high_beta + alpha)

    return VI


def normalize_to_high(data, high=100, axis=-1):
    """

    Parameters
    ----------
    data
    high
    axis

    Returns
    -------

    """
    c_ = np.array(data)
    # Normalize to between 0 and high
    arr = c_.T * np.array([high]) / np.max(np.abs(c_), axis=axis)
    arr = arr.T
    return arr

# def normalize (low =
    # Normalize channels to between -1.0 and +1.0
    # c = np.array(data)
    # c /= np.max(np.abs(c), axis=axis)

    # Normalize image to between 0 and 255
    # c_ = np.array(data)
    # c_ *= (high/np.max(np.abs(c_), axis=a