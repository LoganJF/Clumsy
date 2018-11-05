from __future__ import print_function
from itertools import chain
from collections import deque
import pickle
#import cPickle
#import six
from six.moves import cPickle as cPickle
import sys

__all__ = ['Save_Data', 'Load_Data', 'get_total_size']
try:
    from reprlib import repr
except ImportError:
    print('Could not find rprlib cannot import repr')
    pass

__all__ = ['Save_Data', 'Load_Data', 'total_size']
def Save_Data(Object, path):
    """Saves arbitrary objects using pickling to desired path

    Parameters
    ----------
    Object: object, an arbitrary thing to save
    path: str, file path to Object to
    """
    with open((path), 'wb') as handle:
        pickle.dump(Object, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved successfully, file save as:\n' + path)

def Load_Data(path):
    """Loads objects using pickling

    Parameters
    ----------
    path: str, file path to load from

    Returns
    -------
    loaded_data: object, returned object
    """
    with open(path, 'rb') as handle:
        loaded_data = pickle.load(handle)
        return loaded_data

def sizecheck(object):
    """Check size of object"""
    # Old, should be depreciated/removed
    mydict = {'object': object}
    mydict_as_string = cPickle.dumps(mydict)
    print(sys.getsizeof(mydict_as_string))


def get_total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}


    Parameters
    ----------
    o: Object
    handlers:
    verbose: bool, whether to print out statements.

    Returns
    -------
    size


    ##### Example call #####

    #if __name__ == '__main__':
        #d = dict(a=power_ts)  # a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
        #x = (total_size(d, verbose=True))
        #print(x)  # *1e-9)

    """

    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # User handlers take precedence
    seen = set()  # Track which object id's have already been seen
    default_size = sys.getsizeof(0)  # Estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)