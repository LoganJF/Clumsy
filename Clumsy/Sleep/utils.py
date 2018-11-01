"""Contains utility functions used in analysis of sleep data"""

from logging import getLogger
from numpy import (absolute, arange, argmax, argmin, around, asarray,
                   concatenate, cos, diff, exp, empty, floor, histogram,
                   hstack, insert, invert, log10, logical_and, mean, median,
                   nan, ones, percentile, pi, ptp, real, sqrt, square, std,
                   vstack, where, zeros)
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.signal import (argrelmax, butter, cheby2, filtfilt,
                          fftconvolve, hilbert, periodogram, remez,
                          sosfiltfilt, tukey, tf2zpk)
from scipy.fftpack import next_fast_len
import warnings
import numpy as np
# from ptsa.data.timeseries import TimeSeries
from Clumsy.timeseriesLF import TimeSeriesLF, TimeSeries
from copy import deepcopy
from Clumsy.SpectralAnalysis import HilbertFilter

def transform_signal(dat, method, s_freq=None, method_opt=None, dat2=None, axis=-1):
    """Transform the data using different methods.

    Wrapper around wonambi for use with timeseries, changes in code to vectorize wherever
    possible, and extend 1d functionality to nD
    See: https://wonambi-python.github.io/


    Parameters
    ----------
    dat : TimeSeriesLF or ndarray (dtype='float')
        vector with all the data for one channel
    method : str
        one of 'butter', 'cheby2', 'double_butter', 'morlet', 'morlet_real',
        'hilbert', 'abs', 'smooth', 'gaussian', 'remez', 'wavelet_real',
        'low_butter', 'zscore', 'moving_rms', 'moving_ms'
    s_freq : float defaults as None
        sampling frequency, not necessary if a TimeSeriesLF object is passed
    method_opt : dict
        depends on methods
    dat2 : ndarray(dtype='float')
        second vector with data
    axis : int, defaults as -1
        axis on which to transform the data

    Returns
    -------
    ndarray (dtype='float')
        vector with all the data for one channel
    or
    TimeSeriesLF (dtype='float')
        array with all the data for all channels

    Notes
    -----
    double_butter implements an effective bandpass by applying a highpass,
    followed by a lowpass. This method reduces filter instability, due to
    underlying numerical instability arising from nyquist / freq, at low freq.
    Wavelets pass only absolute values already, it does not make sense to store
    the complex values.

    When using a butterworth filter, it's important to consider stability, if a
    filter is unstable it will explode. To check one can use:
    z, p, k = signal.tf2zpk(b, a)
    assert(np.max(np.abs(p))<=1)
    if the filter is unstable, one can get around this by using a second order
    section (e.g. sosbutter, double_sosbutter)
    see: https://stackoverflow.com/questions/41371915/20hz-20000hz-butterworth-filtering-exploding


    filtfilt is zero-phase filtering, which doesn't shift the signal as it filters. Since the phase is zero at all frequencies, it is also linear-phase.
    Filtering backwards in time requires you to predict the future, so it can't be used in "online" real-life applications, only for offline processing
    of recordings of signals.
    e.g.
    # Applies filter forward and backward in time
    imp_ff = signal.filtfilt(b, a, impulse)

    lfilter is causal forward-in-time filtering only, similar to a real-life electronic filter. It can't be zero-phase. It can be linear-phase
    (symmetrical FIR), but usually isn't. Usually it adds different amounts of delay at different frequencies.
    # Applies filter forward in time twice (for same frequency response)
    imp_lf = signal.lfilter(b, a, signal.lfilter(b, a, impulse))




    Methods
    -------


    butter has parameters: zero-phase filtering
        freq : tuple of float
            low and high values for bandpass
        order : int
            filter order (will be effecively doubled by filtfilt)

    cdemod has parameters:
        freq : float
            carrier frequency for complex demodulation

    cheby2 has parameters:
        freq : tuple of float
            low and high values for bandpass
        order : int
            filter order (will be effecively doubled by filtfilt)

    double_butter has parameters:
        freq : tuple of float
            low and high values for highpass, then lowpass
        order : int
            filter order (will be effecively doubled by filtfilt)

    double_sosbutter has parameters:
        freq : tuple of float
            low and high values for highpass, then lowpass
        order : int
            filter order (will be effecively doubled by filtfilt)

    gaussian has parameters:
        dur : float
            standard deviation of the Gaussian kernel, aka sigma (sec)

    low_butter has parameters:
        freq : float
            Highcut frequency, in Hz
        order : int
            filter order (will be effecively doubled by filtfilt)

    morlet has parameters:
        f0 : float
            center frequency in Hz
        sd : float
            standard deviation of frequency
        dur : float
            window length in number of standard deviations

    wavelet_real has parameters:
        freqs : ndarray
            vector of wavelet frequencies for spindle detection
        dur : float
            duration of the wavelet (sec)
        width : float
            wavelet width
        win : float
            moving average window length (sec) of wavelet convolution

    smooth has parameters:
    NOT IMPLEMENTED YET
        dur : float
            duration of the window (sec)
        win : str
            'flat' or 'triangle' based upon smoothing desired

    moving_rms has parameters:
        dur : float
            duration of the window (sec)

    moving_zscore has parameters:
        dur : float
            duration of the z-score sliding window (sec)

    remez has parameters:
        freq : tuple of float
            low and high values for bandpass
        rolloff : float
            bandwidth, in hertz, between stop and pass frequencies
        dur : float
            dur * s_freq = N, where N is the filter order, a.k.a number of taps

    sosbutter has parameters:
        freq : tuple of float
            low and high values for highpass, then lowpass
        order : int
            filter order (will be effecively doubled by filtfilt)

    tri_smooth has parameters:
        dur : float
            length of convolution window, base of isosceles triangle
    """
    valid_methods = (
    'abs', 'abs2', 'cheby2', 'butter', 'high_butter', 'low_butter', 'sosbutter', 'double_butter', 'double_sosbutter',
    'gaussian', 'hilbert', 'remez', 'moving_covar', 'moving_power_ratio', 'moving_sd', 'moving_zscore', 'moving_rms',
    'moving_ms', 'cdemod', 'morlet', 'smooth', 'wavelet_real')
    not_implemented_yet = ('moving_covar', 'moving_power_ratio', 'moving_sd', 'moving_zscore', 'moving_rms',
                           'moving_ms', 'cdemod', 'morlet', 'smooth', 'wavelet_real')

    is_timeseries = True if issubclass(type(dat), TimeSeries) else False
    if is_timeseries:
        copy = deepcopy(dat)
        if method.lower() != 'hilbert':  # HilbertFilter behavior will fail if dat is an array
            dat = np.array(dat)
        if s_freq is None:
            s_freq = float(copy.samplerate)

    nyquist = s_freq / 2

    # ------> Absolute Value
    if 'abs' == method:
        dat = absolute(dat)

    if 'abs2' == method:
        dat = dat.real ** 2 + dat.imag ** 2

    # ------> Chebyshev type II digital filter
    # TODO: Add in cheby1
    if 'cheby2' == method:
        freq = method_opt['freq']
        N = method_opt['order']
        try:
            Rs = method_opt['Rs']
        except:
            Rs = 40
        Wn = asarray(freq) / nyquist
        b, a = cheby2(N, Rs, Wn, btype='bandpass')
        dat = filtfilt(b, a, dat, axis=axis)

    # ------> Butterworth filter
    if 'butter' in method:
        freq = method_opt['freq']
        N = method_opt['order']
        if method not in ('low_butter', 'high_butter'):
            Wn = asarray(freq) / nyquist
        else:
            Wn = freq / nyquist
        output = 'sos' if 'sos' in method else 'ba'
        correctfiltfilt = sosfiltfilt if 'sos' in method else filtfilt

        def apply_correctfiltfilt(filter_inputs, correctfiltfilt, dat, axis):
            """applies correct filtfilter for sos/ba and unpacks tuple where appropriate"""
            if type(filter_inputs) == tuple:  # * unpacks the tuple
                return correctfiltfilt(*filter_inputs, dat, axis=axis)
            else:
                return correctfiltfilt(filter_inputs, dat, axis=axis)

        # Not sure which line reads better -> Double butter filters
        if method in ('double_butter', 'double_sosbutter'):
            # High pass
            filter_inputs = butter(N, Wn[0], btype='highpass', output=output)
            if output == 'ba':  # Check stability of filter and prompt user to use sos if unstable
                check_stability(b=filter_inputs[0], a=filter_inputs[1])
            dat = apply_correctfiltfilt(filter_inputs, correctfiltfilt, dat, axis)
            # Low pass
            filter_inputs = butter(N, Wn[1], btype='lowpass', output=output)
            if output == 'ba':  # Check stability of filter and prompt user to use sos if unstable
                check_stability(b=filter_inputs[0], a=filter_inputs[1])
            dat = apply_correctfiltfilt(filter_inputs, correctfiltfilt, dat, axis)

        elif method in ('butter', 'high_butter', 'low_butter', 'sosbutter'):

            if 'low' in method:
                btype = 'lowpass'
            elif 'high' in method:
                btype = 'highpass'
            elif method in ('butter', 'sosbutter'):
                btype = 'bandpass'

            filter_inputs = butter(N, Wn, btype=btype, output=output)
            if output == 'ba':  # Check stability of filter and prompt user to use sos if unstable
                check_stability(b=filter_inputs[0], a=filter_inputs[1])
            dat = apply_correctfiltfilt(filter_inputs, correctfiltfilt, dat, axis)

    if 'gaussian' == method:
        truncate = 8
        sigma = method_opt['dur']  # 0.004
        if issubclass(type(dat), TimeSeries):
            dat = gaussian_filter1d(dat, sigma * s_freq,
                                    truncate=truncate, axis=axis, mode='constant')
        else:
            dat = gaussian_filter(dat, sigma)

    if 'hilbert' == method:
        if issubclass(type(dat), TimeSeries):
            hf = HilbertFilter(timeseries=dat)
            return hf.filter()

        else:  # ASSUMES 1D???
            N = len(dat)
            dat = hilbert(dat, N=next_fast_len(N))  # much faster this way
            dat = dat[:N]  # truncate away zero-padding

    if 'remez' == method:
        Fp1, Fp2 = method_opt['freq']
        rolloff = method_opt['rolloff']
        dur = method_opt['dur']

        N = int(s_freq * dur)
        Fs1, Fs2 = Fp1 - rolloff, Fp2 + rolloff
        dens = 20

        bpass = remez(N, [0, Fs1, Fp1, Fp2, Fs2, nyquist], [0, 1, 0],
                      grid_density=dens, fs=s_freq)
        dat = filtfilt(bpass, 1, dat, axis=axis)

    if 'moving' in method:  # This is a stupid implementation change it to rolling array instead of for loops#
        raise NotImplementedError('{} has not been implemented yet'.format(method))
        return

        dur = method_opt['dur']
        step = method_opt['step']
        n_step = int(around(step * s_freq)) if step else None
        n_halfdur = int(s_freq * dur / 2)
        len_dat = len(dat)
        len_out = int(len_dat / n_step) if n_step else len_dat
        last = len_dat - 1
        out = zeros((len_out))

        if 'moving_covar' == method:  #### NEEDS UPDATE ####
            raise NotImplementedError('{} has not been implemented yet'.format(method))
            return

        if 'moving_power_ratio' == method:  #### NEEDS UPDATE ####
            raise NotImplementedError('{} has not been implemented yet'.format(method))
            return
            freq1 = method_opt['freq_narrow']
            freq2 = method_opt['freq_broad']
            fft_dur = method_opt['fft_dur']

        if 'moving_sd' == method:  #### NEEDS UPDATE ####
            raise NotImplementedError('{} has not been implemented yet'.format(method))
            return

        if 'moving_zscore' == method:  #### NEEDS UPDATE ####
            raise NotImplementedError('{} has not been implemented yet'.format(method))
            return
            pcl_range = method_opt['pcl_range']

        if method in ['moving_rms', 'moving_ms']:  #### NEEDS UPDATE ####
            raise NotImplementedError('{} has not been implemented yet'.format(method))
            return
            if method == 'moving_rms':
                pass

    if 'cdemod' == method:  ### Fack this needs to be fixed#### Apply to nD
        carr_freq = method_opt['freq']
        carr_sig = exp(-1j * 2 * pi * carr_freq * arange(0, len(dat)) / s_freq)

        dat = dat * carr_sig

    if 'morlet' == method:  #### NEEDS UPDATE ####
        raise NotImplementedError('{} has not been implemented yet consider using ptsa instead'.format(method))
        return
        f0 = method_opt['f0']
        sd = method_opt['sd']
        dur = method_opt['dur']

        wm = _wmorlet(f0, sd, s_freq, dur)
        dat = absolute(fftconvolve(dat, wm, mode='same'))

    if 'smooth' == method:  #### NEEDS UPDATE ####
        raise NotImplementedError('{} has not been implemented yet'.format(method))
        return

        dur = method_opt['dur']
        win = method_opt['win']

        if 'flat' == win:
            flat = ones(int(dur * s_freq))
            H = flat / sum(flat)
        elif 'triangle' == win:
            T = int(dur * s_freq / 2)
            a = arange(T, 0, -1)

            H = hstack([a[-1:0:-1], a])
            H = H / sum(H)

        dat = fftconvolve(dat, H, mode='same')

    if 'wavelet_real' == method:  #### NEEDS UPDATE ####
        raise NotImplementedError('{} has not been implemented yet consider using ptsa instead'.format(method))
        return
        freqs = method_opt['freqs']
        dur = method_opt['dur']
        width = method_opt['width']
        win = int(method_opt['win'] * s_freq)

        wm = _realwavelets(s_freq, freqs, dur, width)
        tfr = empty((dat.shape[0], wm.shape[0]))
        for i, one_wm in enumerate(wm):
            x = abs(fftconvolve(dat, one_wm, mode='same'))
            tfr[:, i] = fftconvolve(x, tukey(win), mode='same')
        dat = mean(tfr, axis=1)

    if is_timeseries:
        copy.data = dat
        return copy

    return dat