#!/usr/bin/python3


from ephysio.openEphysIO import Loader
from ephysio.vizio import viz

root = 'Desktop/data-20210418_152906' # Must be the root of an OpenEphys recording

x = Loader(root)

streams = x.spikestreams()
stream = streams[0]
print(stream)

dat = x.data(stream)
fs_Hz = x.samplingrate(stream)
chlist = x.channellist(stream)
viz(dat, fs_Hz, chlist)
