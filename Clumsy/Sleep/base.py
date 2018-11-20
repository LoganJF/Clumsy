"""base.py Script for base detector that other detectors inherit from"""
from ptsa.data.timeseries import TimeSeries
from ptsa.data.common import get_axis_index
from scipy.stats import zscore
import numpy as np
import pandas as pd
from Clumsy import rolling_window_full, ButterworthFilter, find_consecutive_data
from numba import jit
import traits.api

__all__ = ['jit_find_containing_intervals', 'BaseDetector']

# -----------> Helper Function
# Use numba to vastly improve the speed of this (x10) through the use of compiling
@jit(nopython=True)
def jit_find_containing_intervals(channel, duration):
    """Find intervals where there is an amplitude and duration crossing

    NOTES: Uses numba to vastly improve the speed of this (x10)
           through the use of just-in-time (jit) compiling of the code
           Use the function find_containing_intervals which acts as a wrapper on this
           rather than this function itself
    ------
    INPUTS:
    ------
    channel: bool array, True if above Threshold else False
    duration: int, duration in samples to threshold by
    ------
    OUTPUTS:
    ------
    (n x 2) array of start, stop indicies
    """
    count = 0
    starts, stops = [], []

    for i, j in enumerate(channel):
        # Crossed the duration threshold!
        if count == duration:
            starts.append(i - duration)

        # Skip zeroth element
        if i == 0:
            continue

        # If the channel crosses amplitude threshold, increment counter
        if j:
            count += 1
            continue

        # Otherwise reset it and it's of a sufficient duration add a stop
        if not j:
            if count >= duration:
                stops.append(i)
            count = 0
            continue

    return starts, stops



# --------------> Base Detection Object
class BaseDetector(traits.api.HasTraits):
    """

    """

    valid = {'spindles': ['rms'],
             'ripples': ['buzsaki'],
             'slow-waves': ['amplitude', 'duration', 'fast'],
             'ied': ['buzsaki']
             }

    time_series = traits.api.Instance(TimeSeries)
    event_type = traits.api.Str
    method = traits.api.Str

    def __init__(self, time_series, event_type=None, method=None):
        """

        Parameters
        ----------
        time_series:TimeSeriesLF/ TimeSeries object,
        event_type
        method
        """
        if not issubclass(type(time_series), TimeSeries):
            raise(TypeError('Please input a time_series object'))

        super(BaseDetector, self).__init__()
        self.time_series = time_series

        #if not event_type in BaseDetector.valid:
            #raise(ValueError('inputted event_type {} not in {}'.format(event_type, BaseDetector.valid)))
        self.event_type = event_type

        #if not method in BaseDetector.valid[event_type]:
            #raise (ValueError('inputted method {} not in {}'.format(method, BaseDetector.valid[event_type])))
        self.method = method
        return

    ###### General Utility Functions #####
    @staticmethod
    def remove_linenoise(ts, harmonic=1, line_noise=60):
        for i in np.arange(1, harmonic+1):
            freqs = ((line_noise*i)-2, (line_noise*i)+2)
            ts = ts.filter_with(ButterworthFilter, freq_range=freqs, order=4, filt_type='stop', filter_output='sos')
        return ts

    def _dim_to_axis(self, dim):
        return get_axis_index(self.time_series, dim)

    def _duration_to_samples(self, duration):
        return self.time_series._TimeSeries__duration_to_samples(duration)

    @staticmethod
    def find_containing_intervals(channel, duration):
        """

        Parameters
        ----------
        channel
        duration

        Returns
        -------
        starts, stops array like contain start and stop of intervals above duration
        """
        return jit_find_containing_intervals(channel=channel, duration=duration)

    ###### Thresholding Functions #####
    def zscore_data(self, dim='time'):
        """

        Parameters
        ----------
        dim: dimension to apply zscore over
        inplace: bool, whether to modify in place or not

        Returns
        -------

        """
        axis = self._dim_to_axis(dim=dim)
        zdata = zscore(self.time_series.data, axis=axis)
        return zdata


    def rootmeansquare(self, window, asteps=None):
        """Applies a moving root mean square of window seconds over timeseries

        Parameters
        ----------
        window: float/int, Time in seconds to apply window over
        asteps: float/int, Time in seconds to allow window non-overlap
                 Aligned at the last axis, new steps for the original array, ie. for
                 creation of non-overlapping windows. (Equivalent to slicing result)

        Returns
        -------

        """
        window_size = self.time_series._TimeSeries__duration_to_samples(window)
        asteps = self.time_series._TimeSeries__duration_to_samples(asteps)
        # Rolling root mean square on the data
        # rolled_data = rolling_window(self.time_series.data ** 2, window_size)
        rolled_data = rolling_window_full(self.time_series.data ** 2, window=window_size, asteps=asteps)
        rms = np.mean(rolled_data, -1)
        rms = np.sqrt(rms)
        return rms

    @staticmethod
    def percentile_threshold(data, threshold=92, axis=1):
        """Apply a percentile based threshold to the data

        Parameters
        ----------
        data: array_like
              data to apply threshold on
        threshold: array_like of float
                   Percentile or sequence of percentiles to compute, which must be between
                   0 and 100 inclusive.
        axis: int
              axis to threshold over

        Returns
        -------
        bool_arr, locations where inputted data is above desired threshold
        """
        # Use percentile thresholding
        percentiles = np.percentile(data, q=threshold, axis=axis)
        return data > percentiles[:, None]

    @staticmethod
    def zscore_threshold(data, threshold=3, axis=-1):
        """Use z-score to find out where data is at n threshold above the standard deviation of the mean

        Parameters
        ----------
        data
        threshold
        axis

        Returns
        -------
        bool_arr, locations where inputted data is above desired threshold
        """
        zdata = zscore(data, axis=axis)
        #mean = np.mean(zdata, axis=axis)
        #std = np.std(zdata, axis=axis)
        #threshold = mean + (std * threshold)
        return zdata > threshold


class IEDDetector(BaseDetector):
    """
    """
    # IED definitions

    methods = {'buzsaki': {'thres_filtered': 2,
                           'thres_unfiltered': 2,
                           'freq_range': (25, 80)}

               }

    def __init__(self, time_series, event_type='ied', method='buzsaki'):
        super(BaseDetector, self).__init__(time_series=time_series, event_type=event_type, method=method)
        self.method_d = IEDDetector.methods[method]
        self.frequency = self.method_d['freq_range']
        return

    def detect_ied(self):
        ts = self.remove_linenoise(self.time_series, harmonic=2, line_noise=60)
        ied = ts.filter_with(ButterworthFilter, freq_range=self.frequency,
                             order=4, filt_type='pass', filter_output='sos')

        thres_filtered = self.method_d['thres_filtered']
        thres_unfiltered = self.method_d['thres_unfiltered']

        # More than n times filtered data -> Keep
        """Keep any time points that exceed the threshold of the filtered data"""
        mean = np.mean(ied.data, 1)
        std = np.std(ied.data, 1)
        thres = mean + (thres_filtered * std)
        thres = thres[:, None]
        boolean_array = ied.data > thres

        # Less than n times baseline -> Remove
        """Discard any time points that are below the threshold of the unfiltered data"""
        mean_raw = np.abs(ts).mean('time')
        std_raw = np.abs(ts).std('time')
        thres_raw = mean_raw.data + (thres_unfiltered * std_raw.data)
        thres_raw = thres_raw[:, None]
        bool_raw = self.time_series.data > thres_raw

        return (bool_raw & boolean_array)

    def detect(self):
        boolean_arr = self.detect_ied()
        dfs = []
        for i, boolean_ch in enumerate(boolean_arr):
            consec = find_consecutive_data(np.where(boolean_ch)[0])
            consec = np.array([x[0] for x in consec])
            consec = consec[np.append(True, np.diff(consec) > self._duration_to_samples(duration=1.))]
            df = pd.DataFrame(consec, columns=['samples'])
            df['channel'] = str(self.time_series.channels[i].data)
            df['event'] = 'IED'
            dfs.append(df)
        return pd.concat(dfs)