"""loadmat.py
A script used to load .mat files
"""
import scipy.io as spio
import scipy

__all__ = ['loadmat']


MAPPER = {'pickle':'pkl', 'numpy':'npy', 'hdf':'h5', 'csv':'csv', 'edf' : 'edf'}


def loadmat(filename):
    """This function should be called instead of direct scipy.io.loadmat as it cures the problem of not properly
    recovering python dictionaries from mat files.

    Calls the function _check_keys to cure all entries which are still mat-objects

    Parameters
    ----------
    filename: str, filename to a matlab path

    Returns
    -------
    dict: dictionary, data from matlab
    """
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """checks if entries in dictionary are mat-objects. If yes _todict is called to change them to nested dictionaries

    Parameters
    ----------
    dict: dictionary

    Returns
    -------
    dict: dictionary
    """
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    """ A recursive function which constructs from matobjects nested dictionaries

    Parameters
    ----------
    matob

    Returns
    -------

    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
