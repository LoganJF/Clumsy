from ptsa.data.timeseries import TimeSeries
from scipy.ndimage.filters import gaussian_filter1d

__all__ = ['gaussian_smooth']

def gaussian_smooth(data, sampling_frequency=None, sigma=0.004, axis=-1, truncate=8):
    '''1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    '''

    if ((sampling_frequency is None) & (type(data) == TimeSeries)):
        sampling_frequency = float(data['samplerate'])

    filtered = gaussian_filter1d(data, sigma * sampling_frequency,
                                 truncate=truncate, axis=axis, mode='constant')

    if type(data) == TimeSeries:
        # Create a copy of timeseries
        copy = TimeSeries(data=filtered, dims=data.dims, coords=data.coords)
        return copy

    return filtered