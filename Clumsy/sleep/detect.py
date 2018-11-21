"""detect.py Module for detection of ripples, spindles, and inter-ictal discharges"""
from ptsa.data.timeseries import TimeSeries
from ptsa.data.common import get_axis_index
from scipy.stats import zscore
from Clumsy import rolling_window_full, ButterworthFilter
from numba import jit
import traits.api
import numpy as np
import pandas as pd
from .base import BaseDetector, jit_find_containing_intervals, find_consecutive_data
from Clumsy import HilbertFilter

__all__ = ['SpindleDetector',
           'IEDDetector',
           'RippleDetector',
           'BaseDetector',
           'jit_find_containing_intervals',
           'find_consecutive_data']



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
        rms: np.array, array of root mean square
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

    @staticmethod
    def threshold_value_from_array(data, threshold):
        return data > threshold

class SpindleDetector(BaseDetector):
    """
    """
    # Spindle definitions
    min_sample, max_sample = .5, 3.
    min_freq, max_freq = 11., 16.
    #duration = (.5, 3)

    methods = {'rms': { 'window' : .2 ,
                        'asteps' : None,
                        'freq' : (11., 16),
                        'threshold' : 92
                      }
               }

    def __init__(self, time_series, event_type='spindle', method='rms', duration=(.5, 3)):


        super(BaseDetector, self).__init__(time_series=time_series, event_type=event_type, method=method)

        self.min_sample = self.time_series._TimeSeries__duration_to_samples(duration[0])
        self.max_sample = self.time_series._TimeSeries__duration_to_samples(duration[1])

        pass

    def rms(self):
        self.rootmeansquare(window=.2, asteps=None, dim='time')

    @staticmethod
    def find_intervals(boolean_arr, min_sample, max_sample, contacts_df=None):
        """Find intervals where there is an amplitude and duration crossing
        ------
        INPUTS:
        ------
        ts: TimeSeriesLF object

        ------
        OUTPUTS:
        ------
        data: list, [ch x array(n x 2)] of start, stop indicies
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
            df = df[df['duration'] < max_sample]
            dataframe.append(df)
        try:
            return pd.concat(dataframe)
        except ValueError as e:
            print(e)
            return dataframe


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
            try:
                consec = np.array([x[0] for x in consec])
            except IndexError as e:
                continue

            consec = consec[np.append(True, np.diff(consec) > self._duration_to_samples(duration=1.))]
            df = pd.DataFrame(consec, columns=['samples'])
            df['channel'] = str(self.time_series.channels[i].data)
            df['event'] = 'IED'
            dfs.append(df)
        try:
            return pd.concat(dfs)
        except ValueError as e:
            return dfs

class RippleDetector(BaseDetector):

    #def __init__(self, time_series):
        #super(RippleDetector, self).__init__(time_series)

    methods = {'buzsaki': {'method': 'zscore',
                           'value threshold': 3.,
                           'freq_range': (80, 250),
                           'duration threshold':.2
                           }

               }

    def __init__(self, time_series, event_type='ripple', method='buzsaki'):
        # TODO: Use *args super()
        super(BaseDetector, self).__init__(time_series=time_series, event_type=event_type, method=method)
        self.method_d = RippleDetector.methods[method]
        self.frequency = self.method_d['freq_range']
        self.order = 4

        return

    @staticmethod
    def _ripple_filter_amplitude_envelope(ts, freq_range=(80,250), order=4):
        """

        Parameters
        ----------
        ts
        freq_range
        order

        Returns
        -------

        """
        # Filter to ripple
        ripple_filt = ts.filter_with(ButterworthFilter, freq_range=freq_range, order=order,
                                     filt_type='pass', filter_output='sos')
        ripple_filt = ripple_filt.filter_with(HilbertFilter)
        ripple_filt.data = np.abs(ripple_filt)
        return ripple_filt

    def ripple_filter_amplitude_envelope(self):
        return  self._ripple_filter_amplitude_envelope(ts=self.time_series, freq_range=self.frequency, order=self.order)

    def detect(self, ripple_filt):
        from Clumsy import chop_intervals, gaussian_smooth, RippleDetector, SpindleDetector
        ripple_filt = self.ripple_filter_amplitude_envelope()
        ripple_filt = gaussian_smooth(ripple_filt)
        boolean_arr = RippleDetector.zscore_threshold(ripple_filt, threshold=3, axis=-1)
        min_sample, max_sample = .2, None
        min_sample = ripple_filt._TimeSeries__duration_to_samples(min_sample)
        df = BaseDetector.find_intervals(boolean_arr, min_sample, max_sample, contacts)


"""
Staresina_2018 -> Hierarchical nesting of slow oscillations, spindles and ripples in the human hippocampus during sleep

https://www.nature.com/articles/nn.4119.pdf

Event detection and extraction. SO, spindle and ripple events were identified independently for each participant and 
channel based on established detection algorithms15,23. 

SOs were detected as follows. First, data were filtered between 0.16–1.25 Hz (two-pass FIR bandpass filter, order = 3 
cycles of the low frequency cut-off), and only  artifact-free data from NREM sleep stages 2–4 were used for event detection. 
Second, all zero-crossings were  determined in the filtered signal, and event duration was determined for SO candidates 
(that is, down-states followed by up-states) as time between two successive positive- to-negative zero-crossings for Cz 
and two successive negative-to-positive zero-crossings for HC, respectively. Events that met the SO duration criteria 
(minimum of 0.8 and maximum of 2 s, 0.5–1.25 Hz) entered the next step. Third, event amplitudes were determined 
for the remaining SO candidates (trough-to-peak amplitude between two positive-to-negative zero crossing for Cz; 
peak-to-trough amplitude between two negative-to-positive zero-crossing for HC). Events that also met the SO amplitude 
criteria (≥75% percentile of SO candidate amplitudes, that is, the 25% of events with the largest amplitudes) were 
considered SOs. Manual validation in a random sampling of the raw EEG data yielded good agreement between hand-scored 
and algorithmically identified SOs, however with greater sensitivity of the automated algorithm for SOs that were less 
pronounced against the background EEG activity. Finally, artifact-free epochs (−2.5 to +2.5 s) time-locked to the SO
down-state in the filtered signal were extracted from the unfiltered raw signal for all events.

Spindles were detected as follows. First, data were filtered between 12–16 Hz (two-pass FIR bandpass filter, order = 3 
cycles of the low frequency cut-off), and only artifact-free data from NREM sleep stages 2–4 were used for event 
detection. Second, the r.m.s. signal was calculated for the filtered signal using a moving average of 200 ms, and the 
spindle amplitude criterion was defined as the 75% percentile of RMS values. Third, Whenever the signal exceeded this 
threshold for more than 0.5 s but less than 3 s (duration criteria) a spindle event was detected. Again, manual 
validation in a random sampling of the raw EEG data yielded good agreement between hand-scored and algorithmically 
identified spindles, however with greater sensitivity of the automated algorithm for spindles that were less 
pronounced against the background EEG activity. Finally, artifact-free epochs (−2.5 to +2.5 s) time-locked to the 
maximum spindle trough in the filtered signal were extracted from the unfiltered raw signal for all events.


Ripples were detected as follows. First, data were filtered between 80–100 Hz (two-pass FIR bandpass filter, order = 3 
cycles of the low frequency cut-off), and only artifact-free data from NREM sleep stages 2–4 were used for event 
detection. Second, the r.m.s. signal was calculated for the filtered signal using a moving average of 20 ms, and the 
ripple amplitude criterion was defined as the 99% percentile of RMS values. Third, whenever the signal exceeded this 
threshold for a minimum of 38 ms (encompassing ~3 cycles at 80 Hz) a ripple event was detected. In addition, we 
required at least three discrete peaks or three discrete troughs to occur in the raw signal segment corresponding to 
the above-threshold RMS segment. This was accomplished by identifying local maxima or minima in the respective raw 
signal segments after applying a one-pass moving average filter
"""