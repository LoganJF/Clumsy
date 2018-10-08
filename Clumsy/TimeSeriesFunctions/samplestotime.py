def sampling_freq_to_minimum_duration(ts, desired_length_in_ms = 20):
    """Computes the minimum window length that corresponds with the desired length in ms"""
    sr = float(ts['samplerate'])
    window_length = int((float(desired_length_in_ms)/1000)/(1/sr))
    return window_length