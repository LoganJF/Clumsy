import numpy as np
from ptsa.data.timeseries import TimeSeries
from scipy.signal import hilbert

def Hilbert(signal):
    """Applies a hilbert transform to the signal (must be a single channel)

    ### IMPORTANT NOTES####
    This function adds zero padding to speed up the processing of the fast
    fourier transformation(FFT) if the length of the signal passed is
    not a power of two (for example a 49999 lengthed signal
    will take orders of magnitude longer than a 50000 lengthed signal to
    compute a Fast fourier transformation on )

    ------
    INPUTS:
    signal: array like or TimeSeries, MUST BE A SINGLE CHANNEL.
    ------
    OUTPUTS:
        signal: TimeSeriesX, a TimeSeries object of shape signal that is hilbert filtered
    """
    padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
    tohilbert = np.hstack((signal, padding))
    result = hilbert(tohilbert)
    result = result[0:len(signal)]

    if type(signal) == TimeSeries:
        # Create a copy of timeseries
        return TimeSeries(data=result, dims=signal.dims, coords=signal.coords)

    return result


def get_amplitude_envelope(signal):
    """Returns that instantaneous amplitude evenlope of the analytic signal
       from the Hilbert transformation
    ------
    INPUTS:
    signal: array like or TimeSeriesX.
    """
    _xarray = np.abs(Hilbert(signal))
    return TimeSeries(_xarray, coords=_xarray.coords)