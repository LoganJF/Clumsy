from glob import glob
from time import time
from Clumsy import TimeSeriesLF
from Clumsy import get_sub_tal
from visbrain import gui
import numpy as np
from cmlreaders import CMLReader
start = time()

paths = glob('/Users/loganfickling/Clumsy/data/*.h5')
paths_done = [
    #'/Users/loganfickling/Clumsy/data/8_17_16_1604_1734_100hz.h5',


]

mp,bp,tal = get_sub_tal("R1207J",'FR1',True)
reader = CMLReader(subject='R1207J', experiment='FR1', session=1, rootdir='/Volumes/rhino')
pairs = reader.load('pairs')
contacts = reader.load('contacts')
electrode_categories = reader.load('electrode_categories')
remove = np.concatenate(list(electrode_categories.values()))
labels = pairs['label']
x = np.array(list(map(lambda s:s.split('-'), labels)))
ch0 = np.in1d(x[:,0], np.intersect1d(x[:,0], remove))
ch1 = np.in1d(x[:,1], np.intersect1d(x[:,1], remove))
healthy_bp = bp[(~ch0 & ~ch1)]


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
