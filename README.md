# ephysio
Python code for reading ephys files

## openEphysIO
This module provides convenient access to continuous and event data saved by OpenEphys. For instance:

    from ephysio import openEphysIO
    import matplotlib.pyplot as plt
    
    ldr = openEphysIO.Loader("/path/to/data")
    strm = ldr.spikestreams()[0] # first probe
    dat = ldr.data(strm) # time x channel map of the data
    fs_Hz = ldr.samplingrate(strm) # sampling rate in Hz
    STIMMARKERCHANNEL = 2
    evts = ldr.nidaqevents()[STIMMARKERCHANNEL] # time stamps of events in channel 2 on the NIDAQ, translated to neuropixel time
    CHANNEL = 100
    STIMNO = 10
    s0 = evts[STIMNO, 0] # Start time of event #10
    ds = np.arange(int(.1*fs_Hz)) # prepare to receive 100 ms of data
    response = dat[s0:s0+ds, CHANNEL] # 100 ms of data following the stimulus 
    plt.plot(ds*1000/fs_Hz, response) # plot data with time in ms on x-axis

Much more documentation is included in the module itself. Most users will want to use the Loader class rather than the various other functions.

## Credits

Developed and tested by DAW and Frank Lanfranchi.
