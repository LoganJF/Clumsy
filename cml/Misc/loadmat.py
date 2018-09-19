import scipy.io as spio

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
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
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
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    """ A recursive function which constructs from matobjects nested dictionaries

    Parameters
    ----------
    matobj

    Returns
    -------

    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
