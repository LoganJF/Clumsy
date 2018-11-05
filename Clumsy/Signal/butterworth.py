"""butterworth.py script used for consctructing various Butterworth Filters"""
import numpy as np
from scipy.signal import (filtfilt, sosfiltfilt, butter, tf2zpk)
try:
    from collections import Sequence
except ImportError as e:
    print('Import error from collections import Sequence')
except Exception as e:
    print(e)
from copy import deepcopy
import warnings
from xarray import DataArray
from ptsa.data.timeseries import TimeSeries
from ptsa.data.common import get_axis_index
from ptsa.data.filters import BaseFilter
import traits.api
from Clumsy.timeseriesLF import TimeSeriesLF

__all__ = ['butterworth_filter', 'ButterworthFilter', 'check_stability']


def check_stability(b, a):
    """utility function that checks the stability of the denom and nom of a filter to ensure it's valid

    Parameters
    ----------
    b: Numerator polynomials of the IIR filter
    a: Denominator polynomials of the IIR filter

    Returns
    -------
    None or a warning if unstable filter

    Example Use:
    --------
    b,a = butter(inputs)
    check_stability(b,a)
    """
    z, p, k = tf2zpk(b, a)
    try:
        assert (np.max(np.abs(p)) <= 1)
    except:
        unstable_msg = "Filter is not stable! np.max(np.abs(p)) must be less than 1 but instead is: {} please use sos approach instead"
        warnings.warn(unstable_msg.format(np.max(np.abs(p))), UserWarning)
    return


def apply_correct_filter(filter_inputs, correct_filter, dat, axis):
    """Utility function to apply correct filter to butterworth

    Parameters
    ----------
    filter_inputs: single argument or argument in tuple
    correct_filter: correct filter to apply either filtfilt or sosfiltfilt
    dat: data to apply filter on
    axis: axis on which to apply the filter

    Returns
    -------
    Filtered data
    """
    """utility applies correct filter for sos/ba and unpacks tuple where appropriate"""
    if type(filter_inputs) == tuple:  # * unpacks the tuple
        return correct_filter(*filter_inputs, dat, axis=axis)
    else:
        return correct_filter(filter_inputs, dat, axis=axis)


def butterworth_filter(data=None, freq=(1, 100), filt_type='bandpass',
                       order=4, sample_rate=None, axis=-1,
                       filter_output='sos', linear_filter=False):
    """
    Function for applying a butterworth filter onto data.

    Parameters
    ----------
    data: TimeSeriesX like
    freq: array-like or int/float depending on filt_type
    filt_type: filter type to use (e.g. 'pass', 'high', 'low', 'highpass',
                'lowpass', 'bandpass', 'band', 'stop', 'bandstop')
    order: int, order of the filter
    sample_rate: int/float, number of samples in a second
    axis: int, axis to filter by default -1
    filter_output: output method for filter, by uses default second order sections
                 valid = ('ba', 'sos')
    linear_filter: bool, NOT IMPLEMENTED YET

    Returns
    -------

    """
    # --------> SANITY CHECKING
    # Make sure the inputted filter type is implemented
    valid_filt_type = ('pass', 'high', 'low', 'highpass', 'lowpass', 'bandpass', 'band', 'stop', 'bandstop')
    if filt_type not in valid_filt_type:
        raise (ValueError('{} was not found in not valid filters\n{}'.format(filt_type, valid_filt_type)))

    # Make sure the high/low pass filters have correct typing
    if (filt_type in ('high', 'low', 'lowpass', 'highpass') and not (type(freq) == float or type(freq) == int)):
        _warning = 'Please use an int or a float for freq input, inputted freq={},type(freq)={}'
        raise (TypeError(_warning.format(freq, type(freq))))

    # If inputted data is a TimeSeries like object, we need to output it as the same type
    is_timeseries = True if issubclass(type(data), TimeSeries) else False
    if is_timeseries:
        # Changing way this is implemented so that it uses less memory...

        #copy = deepcopy(data)
        #data = data.data
        coords = data.coords
        dims = data.dims
        attrs = data.attrs
        name = data.name
        if sample_rate is None:
            sample_rate = float(data['samplerate'].data)

    # Calculate Nyquist frequency to normalize frequency by
    nyquist = sample_rate / 2

    if filt_type in ('low', 'high', 'highpass', 'lowpass'):
        Wn = freq / nyquist
    else:  # All the other can be implemented as two?
        Wn = np.asarray(freq) / nyquist

    # Check which filter we should use, assumes only want second order sections or forward backwards
    correct_filter = sosfiltfilt if filter_output == 'sos' else filtfilt
    if linear_filter:
        # correct_filter = lfilter # Linear phase
        raise NotImplementedError('linear_filt not implemented yet')

    constructed_filter = butter(N=order,  # Order of the filter
                                Wn=Wn,  # Frequency arguments
                                btype=filt_type,  # filter band to use
                                analog=False,  # Digital filter
                                output=filter_output  # ba sos or zpk
                                )
    # Check if we NEED to use second order sections for stability or not
    if filter_output != 'sos':
        check_stability(*constructed_filter)

    # Apply the correct filter and unpacking inputs if necessary
    filtered_data = apply_correct_filter(filter_inputs=constructed_filter,
                                         correct_filter=correct_filter,
                                         dat=data, axis=axis)
    if is_timeseries:
        return TimeSeriesLF(data=filtered_data, coords=coords, dims=dims, attrs=attrs, name=name)
        #copy.data = filtered_data
        #return copy
    return filtered_data


class ButterworthFilter(BaseFilter):
    """Applies Butterworth filter to a time series.
    Keyword Arguments
    -----------------
    timeseries: TimeSeries Data
         TimeSeries object
    freq_range: list-like
       Array [min_freq, max_freq] describing the filter range
    order: int
         Butterworth filter order
    filt_type: str
         String object in: ('pass', 'high', 'low', 'highpass', 'lowpass',
                            'bandpass', 'band', 'stop', 'bandstop')
    filter_output: str
        String object in: ('sos', 'ba') describing whether to use second order sections or not

    .. versionchanged:: 2.0
        Parameter "time_series" was renamed to "timeseries".


    Notes
    -------
    High-order IIR filters can be highly sensitive to quantization of their coefficients,
    and can easily become unstable. This is much less of a problem with first and second-order
    filters; therefore, higher-order filters are typically implemented as serially-cascaded
    biquad sections (and a first-order filter if necessary). The two poles of the biquad filter
    must be inside the unit circle for it to be stable. In general, this is true for all discrete
    filters i.e. all poles must be inside the unit circle in the Z-domain for the filter to be stable.


    """
    order = traits.api.Int
    # freq_range = traits.api.CList(maxlen=2)
    filt_type = traits.api.Str
    design_valid = ('sos', 'ba', 'zpk')
    valid_filt_type = ('pass', 'high', 'low', 'highpass', 'lowpass',
                       'bandpass', 'band', 'stop', 'bandstop')

    def __init__(self, timeseries, freq_range, order=4, filt_type='stop', filter_output='sos'):
        super(ButterworthFilter, self).__init__(timeseries)

        self.freq_range = freq_range
        self.order = order
        self.filt_type = filt_type
        self.filter_output = filter_output

    def filter(self):
        """
        Applies Butterworth filter to input time series and returns filtered TimeSeries object
        Returns
        -------
        filtered: TimeSeries
            The filtered time series
        """
        time_axis_index = get_axis_index(self.timeseries, axis_name='time')
        filtered_array = butterworth_filter(data=self.timeseries,
                                            freq=self.freq_range,
                                            filt_type=self.filt_type,
                                            sample_rate=float(self.timeseries['samplerate']),
                                            axis=time_axis_index,
                                            filter_output=self.filter_output,
                                            order=self.order,
                                            linear_filter=False  # DO NOT CHANGE
                                            )
        if issubclass(type(filtered_array), TimeSeries):
            return filtered_array

        coords_dict = {coord_name: DataArray(coord.copy())
                       for coord_name, coord in list(self.timeseries.coords.items())}
        coords_dict['samplerate'] = self.timeseries['samplerate']
        dims = [dim_name for dim_name in self.timeseries.dims]
        filtered_timeseries = TimeSeries(filtered_array, dims=dims, coords=coords_dict)
        filtered_timeseries.attrs = self.timeseries.attrs.copy()
        return filtered_timeseries