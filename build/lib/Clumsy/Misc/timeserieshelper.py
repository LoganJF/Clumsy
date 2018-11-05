from ptsa.data.timeseries import TimeSeries

def apply_func(timeseries, func, **kwargs):
    """Utility wrapper to apply a function on a timeseries without making it lose it's type, useful for certain
    instances where packages don't necessarily respect the default type of the passed object

    Parameters
    ----------
    timeseries: Timeseries
    func: function, function to apply on timeseries object
    kwargs: function inputs

    Returns
    -------
    Timeseries with function applied to it
    """
    data = func(timeseries, **kwargs)
    return TimeSeries(data=data, dims=timeseries.dims, coords=timeseries.coords)

