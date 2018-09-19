import pickle

def Save_Data(dataframe, savedfilename):
    """
    Allows you to quickly save using pickle, this needs to be gradually phased out
    after updating ptsa to reflect the updated hdf_save function once it's commented
    """
    with open((savedfilename), 'wb') as handle:
        pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved successfully, file save as:\n' + savedfilename)

def Load_Data(savedfilename):
    """
    Allows you to quickly load using pickle, this needs to be gradually phased out
    after updating ptsa to reflect the updated hdf_save/load function once it's documented
    """
    with open(savedfilename, 'rb') as handle:
        loaded_data = pickle.load(handle)
        return loaded_data


from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass


def sizecheck(object):
    import cPickle
    import sys
    mydict = {'object': object}
    mydict_as_string = cPickle.dumps(mydict)
    print(sys.getsizeof(mydict_as_string))


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


##### Example call #####

if __name__ == '__main__':
    #d = dict(a=power_ts)  # a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
    #x = (total_size(d, verbose=True))
    #print(x)  # *1e-9)