# Pizza ptsa stuff
from ptsa.data.timeseries import TimeSeries
from ptsa.data.filters import ButterworthFilter
from pyedflib import EdfReader, EdfWriter
import pyedflib
import numpy as np

__all__ = ['preprocess_raw_edf']

def preprocess_raw_edf(file_path, correct_channels, save_path):
    """Given an edf filepath this will use partial string matches to remove any not matched channels (e.g DC channels etc) and notch filter
    INPUTS
    ------
    file_path: str, path to edf
    correct_channels: np.array, an array of the correct channels for the subject, created using CMLReaders
        e.g. :
        reader = CMLReader(subject='R1207J', experiment="FR1", session=1,
                           localization=0, montage=0)
        ch_df = reader.load('contacts') # Monopolar channels
        correct_channels = np.array(ch_df['label']).view(np.recarray)
    save_path: str, location to save at
    """
    print('Starting to making file: ', save_path)
    f = EdfReader(file_name=file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()

    # Find partial matches in edf raw signals with cmlreaders channel labels
    signal_labels = np.array([label.upper() for label in signal_labels])
    match_indx = []
    for channel in correct_channels:
        match = np.flatnonzero(np.core.defchararray.find(signal_labels, channel) != -1)
        if match.size > 0:
            match_indx.append(match)
    match_indx = np.unique(np.concatenate(match_indx))

    n_valid_signals = match_indx.shape[0]

    # Write edf file
    writer = EdfWriter(save_path, n_channels=n_valid_signals,
                       file_type=pyedflib.FILETYPE_EDFPLUS)

    channel_info = []
    data_list = []

    for i in np.arange(n):

        if i not in match_indx:
            continue

            # if not valid_chs[i]:
            # continue

        ch_dict = {'label': signal_labels[i],
                   'dimension': f.getPhysicalDimension(i),
                   'sample_rate': f.getSampleFrequency(i),
                   'physical_max': f.getPhysicalMaximum(i),
                   'physical_min': f.getPhysicalMinimum(i),
                   'digital_max': 32767.,
                   'digital_min': -32768.,
                   'transducer': f.getTransducer(i),
                   'prefilter': f.getPrefilter(i)}

        channel_info.append(ch_dict)
        data_list.append(f.readSignal(i))

    # Copy old file's annotation
    for annotation in f.read_annotation():
        writer.writeAnnotation(annotation[0], -1, annotation[2])
    writer.setHeader(f.getHeader())

    # Remove line noise notch filter
    sfreq = f.samplefrequency(0)
    time = np.arange(0, data_list[0].shape[0]) / sfreq
    coords = {'channels': signal_labels[match_indx],
              'time': time,
              'samplerate': sfreq}
    ts = TimeSeries.create(data_list, sfreq, coords=coords, dims=['channels', 'time'])
    bf = ButterworthFilter(timeseries=ts, freq_range=[58, 62], order=4, filt_type='stop')
    line_noise_filtered = bf.filter()

    # Create avg reference
    ch_mean = line_noise_filtered.mean('channels')
    avg_ref = line_noise_filtered - ch_mean

    # Save notched global avg as output
    data = [x for x in avg_ref.data]
    assert (len(data) == n_valid_signals)
    writer.setSignalHeaders(channel_info)
    writer.writeSamples(data)

    # Close writer/reader
    writer.close()
    f._close()


if __name__ == '__main__':
    from time import time
    from cmlreaders import CMLReader

    reader = CMLReader(subject='R1207J', experiment="FR1", session=1,
                       localization=0, montage=0)

    df = reader.load('contacts')
    ch_arr = np.array(df['label'])
    path = '/data/eeg/LFSleep/RawData/R1207J/8_17_16/8_17_16_0104_0234.edf'
    save = 'test.edf'
    start = time()
    preprocess_raw_edf(file_path=path, correct_channels=ch_arr, save_path=save)
    print('Total time: ', time() - start)