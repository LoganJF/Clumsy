import numpy as np
import math

__all__ = ['find_nearest', 'round_down']

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