from glob import glob
from time import time
from Clumsy import TimeSeriesLF
from visbrain import gui
import numpy as np

start = time()

paths = glob('/Users/loganfickling/Clumsy/data/*.h5')
paths_done = [
    '/Users/loganfickling/Clumsy/data/8_17_16_1604_1734_100hz.h5',


]

for path in paths:
    if path in paths_done:
        continue
    path = '/Users/loganfickling/Clumsy/data/8_17_16_1604_1734_100hz.h5'
    print('Starting path %s' % path)
    ts = TimeSeriesLF.from_hdf(path)
    data = np.array(ts.data)
    data *= 10**6 # Conversion due to units
    sf = 100 # Could also access in ts
    chan = np.arange(len(ts.channels.data)).astype(str)
    print('Time to start gui', time()-start)
    gui.Sleep(data=data, channels=chan, sf=sf, hypno=None).show()
    print('Close gui on path %s' % path)

    break
