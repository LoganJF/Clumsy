from scipy.signal import (butter, sosfilt)
from ptsa.data.timeseries import TimeSeries

def butterworthfilter(timeseries, freqs, kind='pass', order=5):
    """Applies a bandpass filter on the signal

    Notes: This is stable, meaning it can be used on extremely long chunks of data w/o generating all nans.
    ------
    INPUTS:
    data: TimeSeries; data on which to apply the filter over
    freqs: list [lowcut, high], frequency range to filter over
    kind: str, kind of butterworth filter,
          possible values: 'pass', 'stop'
    order: the order of the filter, by default 5
    """
    nyq = 0.5 * float(timeseries.samplerate.data)
    if type(freqs) == list:
        low = freqs[0] / nyq
        high = freqs[1] / nyq

    if kind in ('pass', 'band', 'bandpass'):
        btype = 'band'
    elif kind in ('stop', 'bandstop'):
        btype = 'bandstop'
    else:
        raise ValueError('unknown value {} for variable kind'.format(kind))

    sos = butter(N=order, Wn=[low, high], analog=False, btype=btype, output='sos')
    data = sosfilt(sos=sos, x=timeseries)
    return TimeSeries(data=data, dims=timeseries.dims, coords=timeseries.coords)