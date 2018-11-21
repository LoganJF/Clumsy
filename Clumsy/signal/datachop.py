# TODO: Imports
def _calc_start_stop_from_offset(offsets, start_time=-1., end_time=2., buffer_time=.5):
    starts = (offsets + ((buffer_time * -1000) + (start_time * 1000))).astype(int)
    stops = (offsets + ((buffer_time * 1000) + (end_time * 1000))).astype(int)
    return starts, stops


def chop_data_from_offsets(timeseries, offsets, start_time, end_time, buffer_time=0.):
    # TODO: Check if time is positive and if it is make it negative so they don't have an empty array returned
    if np.sign(start_time) == np.sign(end_time):
        start_time = -1 * start_time

    start, stop = _calc_start_stop_from_offset(offsets, start_time=start_time,
                                               end_time=end_time, buffer_time=buffer_time)

    time_axis = np.arange(stop[0] - start[0]) / timeseries['samplerate'].data
    # Check if it's symmerical start stop and change it so midpoint in time is 0
    # TODO: Should this be an argument?
    if start_time * -1 == end_time:
        time_axis = time_axis - (time_axis[-1] / 2)
    dat = []
    for _start, _stop in zip(start[:-1], stop[:-1]):
        chopped = timeseries[Ellipsis, slice(_start, _stop)]
        chopped['time'].data = time_axis
        dat.append(chopped)
    data = TimeSeriesLF.concat(dat, 'events')
    data['buffer_time'] = buffer_time
    return data

# TODO: Put these in a better place
def minimum_time_from_frequency_ncycles(frequency, cycles=5):
    return cycles / frequency


def numbers_between(arr, low, high, values=False):
    if values:
        return arr[numbers_between(arr, low, high, False)]
    return np.logical_and(low <= arr, arr <= high)


def dataframe_from_connectivity_matrix(connectivity_matrix, frequency_bands, frequency_selection):
    dat = conectivity_matrix[:, :, numbers_between(frequency_bands, *frequency_selection)]
    return pd.DataFrame(np.mean(dat, -1))


def group_dataframe_from_rois(df, rois):
    """Given a dataframe N x N, returns mean M x M matrix of grouped mean rois

    INPUTS
    -------
    df: pd.DataFrame, shape = N x N
    rois: np.array like, shape N"""
    rois = pd.Series(rois)
    rois_df = df.groupby(by=rois, axis=0).mean().groupby(by=rois, axis=1).mean()
    return rois_df