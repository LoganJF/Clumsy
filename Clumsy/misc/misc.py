import numpy as np
import math

__all__ = ['fill_between_stderr','find_nearest', 'round_down', 'list_of_dict_to_dict']


def fill_between_stderr(arr):
    """
    arr: array of all subjects, axis 0 = subjects
    arr_tstat: array of plotting falls we'll use for the fillbetween
    """
    mean = np.nanmean(arr,axis=0)
    std_err_mean = np.nanstd(arr,axis=0) / ( np.sqrt( len(arr) ) )
    y1, y2 = np.array( (mean+std_err_mean) ), np.array( (mean-std_err_mean) )
    return y1, y2

def find_nearest(array, value, return_index_not_value=True, is_sorted=True):
    """Given an array and a value, returns either the index or value of the nearest match

    Parameters
    ----------
    array: np.array, array of values to check for matches
    value: int/float, value to find the closest match to
    return_index_not_value: bool, whether to return the index(True) or the value (False)
        of the found match
    is_sorted: bool, whether the array is sorted in order of values
    Returns
    -------
    Either the index or value of the nearest match
    """
    if is_sorted:
        idx = np.searchsorted(array, value, side="left")

        if ((idx > 0) and (idx == len(array)) or (math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx]))):

            if not return_index_not_value:
                return array[idx - 1]

            if return_index_not_value:
                return idx - 1

        else:

            if not return_index_not_value:
                return array[idx]

            if return_index_not_value:
                return idx

    elif not is_sorted:
        idx = (np.abs(array - value)).argmin()

        if not return_index_not_value:
            return array[idx]

        if return_index_not_value:
            return idx


def round_down(n):
    """"Rounds number n to nearest 100th place

    Parameters
    ----------
    n: int/float, a number to round down

    Returns
    -------
    n: float, n rounded to nearest 100th place
    """
    return float(int(n / 100) * 100)

def list_of_dict_to_dict(d_list):
    '''
    Take a list of dictionaries and turn it into one dictionary
    ------
    INPUTS:
    d_list: a list of dictionaries
    ------
    RETURNS:
    d: a dictionary that's a merge of the values of the dictionaries in d_list
    '''
    d = {}
    # Go through dicts, add values into d
    for dictionaries in d_list:
        for keys, values in enumerate(dictionaries):
            if values not in d:
                d[values] = []
            if values in d:
                d[values].append(dictionaries[values])
    return d