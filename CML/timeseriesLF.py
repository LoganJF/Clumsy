from ptsa.data.timeseries import TimeSeries

try:
    from mne.io import read_raw_edf
except ImportError:
    read_raw_edf = None
# Turn off stupid mne RunTimeWarnings for EDF+
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
from datetime import datetime, date
from pyedflib import EdfWriter
import xarray as xr
import numpy as np
from IPython.display import display
import pyedflib

__all__ = ['TimeSeriesLF']

class TimeSeriesLF(TimeSeries):
    """A thin wrapper around :class:`xr.DataArray` for dealing with time series
    data and timeseries.
    Note that xarray internals prevent us from overriding the constructor which
    leads to some awkwardness: you must pass coords as a dict with a
    ``samplerate`` entry.
    Parameters
    ----------
    data : array-like
        Time series data
    coords : dict-like
        Coordinate arrays. This must contain at least a ``samplerate``
        coordinate.
    dims : array-like
        Dimension labels
    name : str
        Name of the time series
    attrs : dict
        Dictionary of arbitrary metadata
    encoding : dict
    fastpath : bool
        Not used, but required when subclassing :class:`xr.DataArray`.
    Raises
    ------
    AssertionError
        When ``samplerate`` is not present in ``coords``.
    See also
    --------
    xr.DataArray : Base class
    """

    def __init__(self, data, coords, dims=None, name=None,
                 attrs=None, encoding=None, fastpath=False):
        assert 'samplerate' in coords
        super(TimeSeriesLF, self).__init__(data=data, coords=coords, dims=dims,
                                           name=name, attrs=attrs, encoding=encoding,
                                           fastpath=fastpath)

    """Start Logan Monkey Patching"""

    @classmethod
    def from_edf(cls, filename, correct_channels=None, *args, **kwargs):
        """Takes a file located at filename and returns it as a TimeSeriesX object

        FIXME: automate the generation of 'events' based upon raw signal for sleep data

        ------
        INPUTS
        filename: str, path to the edf file, e.g. '/Volumes/rhino/home2/loganf/data.edf'

        ------
        OUTPUTS
        ts: TimeSeriesX

        """
        if read_raw_edf is None:
            raise RuntimeError("You must install mne to load from EDF")
        print('Extracting edf signal from {}'.format(filename))
        # Use MNE to read in raw edf file
        raw_data = read_raw_edf((filename), preload=False, verbose='ERROR', stim_channel=None,
                                *args, **kwargs)

        # Handling for clinical copying bizzare filenames
        if correct_channels is not None:
            matched = cls.get_eeg_ch_matches(raw_data.ch_names, correct_channels)
            all_chs = np.arange(len(raw_data.ch_names))
            bad_indx = [x for x in all_chs if x not in matched]
            exclude = (list(np.array(raw_data.ch_names)[bad_indx]))
            raw_data = read_raw_edf(filename, stim_channel=None, exclude=exclude, preload=True,
                                    verbose='ERROR', *args, **kwargs)

        # Handling for normal things
        elif correct_channels is None:
            # Mne defaults an empty channel for stim or annotations this will make sure it's removed
            # By default it's assigned to the last channel....
            if raw_data.ch_names[-1] in ('EDF Annotations', 'STI 014'):
                raw_data = read_raw_edf(filename, stim_channel=None, exclude=raw_data.ch_names[-1], preload=True,
                                        verbose='ERROR', *args, **kwargs)
            else:
                raw_data.load_data()

        # Get data, channel labels, sampling freq
        data, chs, sfreq = raw_data._data, raw_data.ch_names, np.round(raw_data.info['sfreq'])
        # Construct time from number of points and sampling freq
        time = np.arange(0, data.shape[-1]) / sfreq
        # Set coords for timeseries
        coords = {'channels': chs, "time": time, "samplerate": sfreq}
        # Create TimeSeriesX object
        ts = cls.create(data, sfreq, coords=coords,
                        dims=['channels', 'time'])
        print('Ready.')
        return ts

    @classmethod
    def concat(cls, ts_list, dim='events', *args, **kwargs):
        """Concatenates a list of TimeSeriesX objects along a dimension

        FIX ME: Change to check if any dim is np.recarray and reset any that are.
        -----
        INPUTS
        ts_list: list, a list of time_series seperated, e.g. list of sessions
        dim: str, dimension to concatenate over, you can also choose a new name (e.g. 'subjects'
             across all a list of all subjects). By default tries to do events
        -----
        OUTPUTS
        ts: TimeSeriesX Object, a functional timeseries object with indexable events
        """

        # Check if events is in the dims
        if all(['events' in y for y in [x.dims for x in ts_list]]):
            # Extract events before overwriting them
            evs = np.concatenate([x.events.data for x in ts_list]).view(np.recarray)
            ts = xr.concat(objs=ts_list, dim=dim, *args, **kwargs)
            ts['events'] = evs  # Reset the events to the correct np.recarray
            return ts

        # Try to just concat normally, if that doesn't work show them why.
        else:
            try:
                ts = xr.concat(objs=ts_list, dim=dim, *args, **kwargs)
                return ts
            except:
                print(ts_list[0].dims)
                print('There needs to be an "events" dim for each of the TimeSeriesX in the passed list')
                assert all(['events' in y for y in [x.dims for x in ts_list]])

    def to_edf(self, save_path, annotations=None, header=None):
        valid = np.array(['channel', 'channels', 'bipolar_pair', 'bipolar_pairs'])
        # Find out what the potential dimension name for the contacts is
        label = valid[np.where(np.in1d(valid, np.intersect1d(np.array(self.dims), valid)))][0]
        try:
            assert label == self.dims[0]
        except AssertionError:
            raise AssertionError('Please ensure that the first dimension of the timeseries is {}'.format(label))

        n_valid_signals = self[label].shape[0]

        writer = EdfWriter(save_path, n_channels=n_valid_signals,
                           file_type=pyedflib.FILETYPE_EDFPLUS)

        channel_info, data_list = [], []
        physical_maximums = self.max('time')
        physical_minimums = self.min('time')

        for i in np.arange(n_valid_signals):
            ch_dict = {'label': str(self[label][i].data),
                       'dimension': 'uV',
                       'sample_rate': float(self.samplerate.data),
                       'physical_max': float(physical_maximums[i]),
                       'physical_min': float(physical_minimums[i]),
                       'digital_max': 32767.,
                       'digital_min': -32768.,
                       'transducer': '',
                       'prefilter': ''}

            channel_info.append(ch_dict)
            data_list.append(self[i].data)

        # Copy old file's annotation
        if annotations is not None:
            for annotation in annotations:
                writer.writeAnnotation(annotation[0], -1, annotation[2])

        # Create a default header if none is passed
        if header is None:
            header = {'technician': '',
                      'recording_additional': '',
                      'patientname': 'No Name',
                      'patient_additional': '',
                      'patientcode': '0',
                      'equipment': 'NKC-EEG-1200A V01.00',
                      'admincode': '',
                      'gender': '',
                      'startdate': datetime(2017, 8, 17, 1, 4, 43),
                      'birthdate': date(1951, 8, 2)}
        writer.setHeader(header)

        writer.setSignalHeaders(channel_info)
        writer.writeSamples(data_list)

        # Close writer/reader
        writer.close()

    @staticmethod
    def get_eeg_ch_matches(signal_labels, contacts_label):
        """ This is used in cases where we want to load data raw off the clinical record and replace bad names with good names
        f = EdfReader(file_name=file_path)
        signal_labels = f.getSignalLabels()
        reader = CMLReader(subject='R1207J', experiment="FR1", session=1,
                           localization=0, montage=0)
        df = reader.load('contacts')
        contacts_label = np.array(df['label'])
        """
        # Find partial matches in edf raw signals with cmlreaders channel labels
        signal_labels = np.array([label.upper() for label in signal_labels])
        match_indx = []
        for channel in contacts_label:
            match = np.flatnonzero(np.core.defchararray.find(signal_labels, channel) != -1)
            if match.size > 0:
                match_indx.append(match)
        match_indx = np.unique(np.concatenate(match_indx))
        return match_indx

    """End Logan Monkey Patching"""

    def fix_ch_names(self, verbose=False):
        valid = np.array(['channel', 'channels', 'bipolar_pair', 'bipolar_pairs'])
        # Find out what the potential dimension name for the contacts is
        label = valid[np.where(np.in1d(valid, np.intersect1d(np.array(self.dims), valid)))][0]
        signal_labels = self[label].data
        format_strings = np.array(list(map(lambda s: s.split(' ')[-1].replace('-Ref', '').upper(),
                                           np.array(signal_labels))))
        if verbose:
            print('Replacing Old Channel labels with new channel labels')
            display(list(zip(self[label].data, format_strings)))

        self[label].data = format_strings
        return