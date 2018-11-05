from visbrain import gui
import numpy as np
from time import time
from ptsa.data.TimeSeries import TimeSeries
start = time()
path = '/Users/loganfickling/Clumsy/R1207J_8_16_16_2334_0104_13after_100hz.h5'
ts = TimeSeries.from_hdf(path)
for x in ts:
    s2=time()
    print('Loading Time: ', s2-start)
    # Format input for Sleep GUI
    data = np.array(x.data)
    sf = 100
    #chan = np.array(ts.bipolar_pairs.data)
    chan = np.arange(len(x.channels.data)).astype(str)

    gui.Sleep(data=data, channels=chan, sf=sf, hypno_file=None).show()
    print(time()-s2)