"""base.py Script for base detector that other detectors inherit from"""
from ptsa.data.timeseries import TimeSeries
from ptsa.data.common import get_axis_index
from scipy.stats import zscore
import numpy as np
import pandas as pd
from Clumsy import rolling_window_full, ButterworthFilter
from numba import jit
import traits.api

__all__ = ['jit_find_containing_intervals', 'BaseDetector', 'find_consecutive_data']

# -----------> Helper Functions
def find_consecutive_data(data, stepsize=1):
    """Splits Data into a list of arrays where there is more than a single step of 1

    Parameters
    ----------
    data: np.array of zeroes and ones, the data to split
    stepsize: int, by default 1

    Returns
    -------
    list of split data
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


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

    @staticmethod
    def find_intervals(boolean_arr, min_sample, max_sample, contacts_df=None):
        """Find intervals where there is an amplitude and duration crossing

        Parameters
        ----------
        boolean_arr: np.array dtype=bool, shape = n chs x samples
                     Array indicating True if valid amplitude threshold or False if not
        min_sample: minimum time in samples to count as a valid duration interval
        max_sample: int or None,
                    minimum time in samples to count as a valid duration interval
        contacts_df: CMLReader contacts data frame

        Returns
        -------

        """

        try:
            fs_roi = contacts_df['ind.region']
            stein_roi = contacts_df['stein.region']
            hemi = ['left' if x == -1 else 'right' for x in np.sign(contacts_df['ind.x'])]
            channels = contacts_df['label']
        except AttributeError:
            fs_roi = ''
            stein_roi = ''
            hemi = ''
            channels = ''
        except Exception as e:
            print(e)
            fs_roi = ''
            stein_roi = ''
            hemi = ''
            channels = ''

        dataframe = []
        for index, ch_arr in enumerate(boolean_arr):
            start, stops = jit_find_containing_intervals(ch_arr, min_sample)
            df = pd.DataFrame(start, columns=['start'])
            try:
                df['stop'] = stops
            except ValueError as e:
                if len(start) != len(stops):  # Should occur when none are found
                    continue
                print(e)
            df['duration'] = df['stop'] - df['start']
            df['channel'] = channels[index]
            df['fs region'] = fs_roi[index]
            df['hemisphere'] = hemi[index]
            df['stein region'] = stein_roi[index]
            if max_sample is not None:
                df = df[df['duration'] < max_sample]
            dataframe.append(df)
        try:
            return pd.concat(dataframe)
        except ValueError as e:
            print(e)
            return dataframe

    ###### Thresholding Functions #####

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