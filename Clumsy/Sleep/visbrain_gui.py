from glob import glob
from time import time
from Clumsy import TimeSeriesLF
from visbrain import gui
import numpy as np

start = time()

paths = glob('data/*.h5')
#paths_done = glob('data/*.txt')
#list(map(os.path.basename, paths_done))
#paths = [x for x in glob(paths) if x.split('.')[-1] != ]
for path in paths:
    print('Starting path %s' % path)
    ts = TimeSeriesLF.from_hdf(path)
    data = np.array(ts.data)
    sf = 100
    # chan = np.array(ts.bipolar_pairs.data)
    chan = np.arange(len(ts.channels.data)).astype(str)
    print('Time to start gui', time()-start)
    gui.Sleep(data=data, channels=chan, sf=sf, hypno=None).show()
    break
