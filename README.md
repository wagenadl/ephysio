# ephysio
Python code for reading ephys files

## openEphysIO
This module provides convenient access to continuous and event data saved by OpenEphys. For instance:

    from ephysio import openEphysIO
    import matplotlib.pyplot as plt
    
    ldr = openEphysIO.Loader("/path/to/data")
    
    # first probe
    strm = ldr.spikestreams()[0] 
    
    # time x channel map of the data
    dat = ldr.data(strm) 
    
    # sampling rate in Hz
    fs_Hz = ldr.samplingrate(strm) 
    STIMMARKERCHANNEL = 2
    
    # time stamps of events in channel 2 on the NIDAQ, translated to neuropixel time
    evts = ldr.nidaqevents()[STIMMARKERCHANNEL] 
    CHANNEL = 100
    STIMNO = 10
    
    # Start time of event #10
    s0 = evts[STIMNO, 0] 
    
    # prepare to receive 100 ms of data
    ds = np.arange(int(.1*fs_Hz)) 
    
    # 100 ms of data following the stimulus 
    response = dat[s0:s0+ds, CHANNEL] 
    
    # plot data with time in ms on x-axis
    plt.plot(ds*1000/fs_Hz, response) 

Much more documentation is included in the module itself. Most users will want to use the Loader class rather than the various other functions.

## Credits

Developed and tested by DAW and Frank Lanfranchi.
