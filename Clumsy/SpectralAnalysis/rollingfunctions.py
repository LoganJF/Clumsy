import numpy as np
from ptsa.data.timeseries import TimeSeries

__all__ = [
    'reorder_dims',
    'xarray_rolling_average',
    'rolling_window_full',
    'rolling_window',
    'rolling_apply'
]
def reorder_dims(darray, dim1, dim2):
    """Interchange two dimensions of a DataArray in a similar way as numpy's swap_axes

    Parameters
    ---------
    darray: Xarray.DataArray/TimeSeriesX
    dim1: str, the first dim we want to swap
    dim2: str, the second dim we want to swap

    Returns
    ---------
    transposed darray

    Example Usage
    -------------
    # Swap the timeseries object's first dim and time dim
    check_first_dim = timeseries.dims[0]
    timeseries_swapped_time_dim = reorder_dims(timeseries, dim1=check_first_dim, dim2='time')

    See
    -------
    https://github.com/pydata/xarray/issues/1081, rafa-guedes comment on Aug 21, 2017
    """
    dims = list(darray.dims)
    assert set([dim1, dim2]).issubset(dims), 'dim1 and dim2 must be existing dimensions in darray'
    ind1, ind2 = dims.index(dim1), dims.index(dim2)
    dims[ind2], dims[ind1] = dims[ind1], dims[ind2]
    return darray.transpose(*dims)


def xarray_rolling_average(timeseries, desired_window, desired_step):
    """Computes a rolling average analogous to sliding_mean_fast using xarray.rolling given a desired window, and step

    Parameters
    ----------
    timeseries: TimeSeriesX, time series we which to average into temporal windows.
                Assumes time dim is explicitly labeled time
    desired_window: float, time in seconds of the desired window (e.g. 500ms/.5s is .5)
    desired_step: float, time in seconds of the desired step between windows (e.g. 10ms is .01)

    Returns
    -------
    timeseries_rolled: TimeSeriesX, time series we inputted but with time dim now representing the midpoint of that time
    point averaged across desired_window seconds.
    """
    first_dim = timeseries.dims[0]
    dims_reordered = False
    if first_dim != 'time':
        # Transpose it so that time is on the first axis for operations
        dims_reordered = True
        timeseries = reorder_dims(darray=timeseries, dim1=first_dim, dim2='time')

    # Use sample rate to determine indexing for windows and steps
    samplerate = 1 / timeseries['samplerate'].data
    timeseries_window_length = int(desired_window / samplerate)
    timeseries_step_length = int(desired_step / samplerate)

    # Using default xarray.DataArray.rolling to compute a mean
    rolling_average = timeseries.rolling(time=timeseries_window_length, center=True).mean()

    # Data is only valid from half the window after start until half before end; e.g. removing nans
    timeseries_rolled = rolling_average[int(timeseries_window_length / 2):int(-timeseries_window_length / 2)]

    # Return windows that correspond with our desired step
    timeseries_rolled = timeseries_rolled[::timeseries_step_length]
    if dims_reordered:
        timeseries_rolled = reorder_dims(darray=timeseries_rolled,
                                         dim1='time',
                                         dim2=first_dim)
    return TimeSeries(data=timeseries_rolled.data, dims=timeseries_rolled.dims, coords=timeseries_rolled.coords)

def rolling_window_full(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.

    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.

    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.

    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])

    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])

    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)

    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])

    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])


    see:
    https://gist.github.com/seberg/3866040
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

def rolling_apply(fun, a, w):
    """

    Parameters
    ----------
    fun
    a
    w

    Returns
    -------

    """
    # http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i - w + 1):i + 1])
    return r


def rolling_window(a, window):
    """

    Parameters
    ----------
    a
    window

    Returns
    -------

    """
    # http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)