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

from xarray import DataArray
from numpy import ceil, log2
from ptsa.data.timeseries import TimeSeries
from ptsa.data.common import get_axis_index
from ptsa.data.filters import BaseFilter, ButterworthFilter
from scipy.signal import hilbert
from copy import deepcopy
import xarray as xr

__all__ = ['HilbertFilter']


class HilbertFilter(BaseFilter):
    """Applies Hilbert filter to a time series.

    Keyword Arguments
    -----------------

    timeseries
         TimeSeries object
    """

    def __init__(self, timeseries):
        """

        Parameters
        ----------
        timeseries
        """
        super(HilbertFilter, self).__init__(timeseries)

    def filter(self):
        """

        Returns
        -------

        """
        time_axis = get_axis_index(self.timeseries, 'time')
        # Zero pad to speed up calculation
        padding = int(2 ** ceil(log2(self.timeseries.shape[time_axis])))  # - signal.shape[time_axis])
        output = hilbert(self.timeseries, N=padding, axis=time_axis)
        # Find out where we need to remove the padding since scipy doesn't do it by default
        valid_pad = (Ellipsis, slice(self.timeseries.shape[time_axis]))
        returned_signal = output[valid_pad]
        # Create a timeseries for the user
        coords_dict = {coord_name: DataArray(coord.copy()) for coord_name, coord in
                       list(self.timeseries.coords.items())}
        coords_dict['samplerate'] = self.timeseries['samplerate']
        dims = [dim_name for dim_name in self.timeseries.dims]
        filtered_timeseries = TimeSeries(returned_signal, dims=dims, coords=coords_dict)
        filtered_timeseries.attrs = self.timeseries.attrs.copy()
        return filtered_timeseries

    def filter_dict(self, dictionary):
        """

        Parameters
        ----------
        dictionary

        Returns
        -------

        """
        data = []
        self._timeseries = deepcopy(self.timeseries)
        for keys, value in enumerate(dictionary):
            bf = ButterworthFilter(timeseries=self._timeseries,
                                   freq_range=dictionary[value],
                                   filt_type='pass',
                                   order=4)
            self.timeseries = bf.filter()
            data.append(self.filter())
            #if keys == 0:
                #data = self.filter()#data[0]
                #data = deepcopy(self.timeseries)
            #else:
                #output = self.filter()
                #data = data.append(output, dim='frequency')
        concat = xr.concat(data, dim='frequency')
        concat['frequency'] = list(dictionary.keys())
        return concat