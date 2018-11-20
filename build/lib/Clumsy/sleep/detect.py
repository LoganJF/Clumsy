"""detect.py Module for detection of ripples, spindles, and inter-ictal discharges"""
import numpy as np
import pandas as pd
from Clumsy import ButterworthFilter
from .base import BaseDetector, jit_find_containing_intervals, find_consecutive_data

__all__ = ['SpindleDetector', 'IEDDetector', 'RippleDetector']

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
        # if channels is None:
        # channels = np.arange(boolean_arr.shape[0])

        fs_roi = contacts_df['ind.region']
        stein_roi = contacts_df['stein.region']
        hemi = ['left' if x == -1 else 'right' for x in np.sign(contacts_df['ind.x'])]
        channels = contacts_df['label']

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

    def __init__(self, time_series):
        super(RippleDetector, self).__init__(time_series)
